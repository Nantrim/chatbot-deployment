import threading
import slack
from flask import Flask
from slackeventsapi import SlackEventAdapter
import pandas as pd
import logging
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
from huggingface_hub import InferenceClient
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import openai
import gradio as gr

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


openai.api_key = "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"

SLACK_TOKEN = "xoxb-210779867969-5630341758432-TSX3oejptCbjURWPBEdQV65Z"
SIGNING_SECRET = "6ae856b7a2f0dbe5b390d33190d7cebc"
SHEET_ID = "1eRhc9BL_UlMu22ZdyCbJnewnWexwCizt-Ka3Yt6WZOg"
SHEET_NAME = "CTAQNA"
url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
# This is a hybrid model that uses parametric (pre-trained gpt) and non-parametric (retrieval-based) memory

# Retrieval Augmented Generation
# https://arxiv.org/abs/2005.11401


def writeAnswer(question, answer):
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

    credentials = Credentials.from_service_account_file('gauth.json', scopes=scopes)
    gc = gspread.authorize(credentials)
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    # open a google sheet
    gs = gc.open_by_url('https://docs.google.com/spreadsheets/d/1eRhc9BL_UlMu22ZdyCbJnewnWexwCizt-Ka3Yt6WZOg/edit?pli=1#gid=0')
    # select a work sheet from its name
    worksheet1 = gs.worksheet('Sheet1')
    df = pd.DataFrame({'question': [question], 'answer': [answer]})
    df_values = df.values.tolist()
    gs.values_append('Sheet1', {'valueInputOption': 'RAW'}, {'values': df_values})


document_store = InMemoryDocumentStore()
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=True,
    scale_score=False,
)


def createPipe():
    df = pd.read_csv(url)
    df.fillna(value="", inplace=True)
    df["question"] = df["question"].apply(lambda x: x.strip())
    print(df.head())
    questions = list(df["question"].values)
    df["embedding"] = retriever.embed_queries(queries=questions).tolist()
    df = df.rename(columns={"question": "content"})
    docs_to_index = df.to_dict(orient="records")
    document_store.write_documents(docs_to_index)
    return FAQPipeline(retriever=retriever)


app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, '/slack/events', app)


client = slack.WebClient(token=SLACK_TOKEN)
# client.chat_postMessage(channel='#bot-test',text='Hello')

# prediction = createPipe().run(query="what does ctaboi stand for?", params={"Retriever": {"top_k": 3}})
# print(prediction.get('answers')[0].meta.get('query'))


@slack_event_adapter.on('message')
def message(payload):
    print(payload)
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')

    # Bot kept timing out - this is the solution
    x = threading.Thread(
        target=processing,
        args=(event, channel_id, user_id, text)
    )
    x.start()
    return "Processing information... please wait"


def processing(event, channel_id, user_id, text):
    if text.startswith("!question"):
        # Check if question exists in the spreadsheet
        qna_df = pd.read_csv(url)
        question = text[10:]

        if question.lower() in qna_df['question'].str.lower().tolist():
            # Question exists in the spreadsheet, retrieve answer from the spreadsheet
            answer = qna_df[qna_df['question'].str.lower() == question.lower()]['answer'].values[0]
            client.chat_postMessage(channel=channel_id, text="Answer (from spreadsheet): " + answer)
            return

        else:
            # Question not found in the spreadsheet, proceed with retrieval-based and parametric models
            # Non-parametric memory
            prediction = createPipe().run(query=question, params={"Retriever": {"top_k": 1}})
            similar_question = prediction.get('answers')[0].meta.get('query')
            client.chat_postMessage(channel=channel_id, text="Most similar question found: \n" + similar_question)
            client.chat_postMessage(
                channel=channel_id,
                text="If this does not match your question, please update my database with the proper question and answer.\n"
                "You can do this by typing your question and answer in the format \"!qna question/answer\""
            )

            # Add context to the prompt
            context = f"Regarding the corporate transparency act (CTA), businesses must disclose beneficial owner information (BOI) to the Financial Crimes Enforcement Network (FINCEN)."
            prompt = f"{question}\n{context}\n\n{prediction.get('answers')[0].answer}"

            # Generate response from the GPT-3.5 model for the given question
            response = openai.Completion.create(
                engine="text-davinci-004",  # Specify the upgraded GPT model version
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.2
                ).choices[0].text
            client.chat_postMessage(channel=channel_id, text="Answer: " + response)
            return

    if text.startswith("!qna"):
        qna = text[5:].split('/', 1)
        writeAnswer(qna[0], qna[1])
        client.chat_postMessage(
            channel=channel_id,
            text="Successfully recorded as \nQuestion: " + qna[0] + "\nAnswer: " + qna[1]
        )
        return


if __name__ == "__main__":
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Set Flask timeout to be infinite
    app.run(debug=True)