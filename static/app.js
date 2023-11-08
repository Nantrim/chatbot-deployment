class Chatbox{
	constructor(){
		this.args = {
			openButton: document.querySelector(selectors'.chatbox__button'),
			chatBox: document.querySelector(selectors'.chatbox__support'),
			sendButton: document.querySelector(selectors'.send__button')
		}
	
		this.state = false;
		this.message = [];
}

	display(){
		const {openButton, chatBox, sendButton} = this.args;
		
		openButton.addEventListener(type'click', listener() => this.toggleState(chatBox))
		
		sendButton.addEventListener(type'click', listener() => this.onSendButton(chatBox))
		
		const node = chatBox.querySelector(selectors'input');
		node.addEventListener(type'keyup',listener({key : string})) => {
			if (key === "Enter"){
				this.onSendButton(chatBox)
				
			}
		})
	}
				
}				