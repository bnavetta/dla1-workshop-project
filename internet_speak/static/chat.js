function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * i);
        const atJ = arr[j];
        const atI = arr[i];
        arr[j] = atI;
        arr[i] = atJ;
    }
}

function sendMessage(message) {
    return fetch(`/message?message=${encodeURIComponent(message)}`)
        .then(response => {
            if (response.status >= 400) {
                let err = new Error(`HTTP ${response.status} error sending "${message}`);
                err.response = response;
                throw err;
            } else {
                return response.json().then(json => json.replies);
            }
        });
}

const names = {
    feels: 'i_hate_mondays',
    computers: 'cicero42',
    wreck: 'hungrydonut'
};

const app = new Vue({
    el: '#app',
    data: {
        messages: [],
        input: ''
    },
    methods: {
        send() {
            this.messages.push({ user: 'MysteriousStranger99', text: this.input, author: 'you' });
            sendMessage(this.input).then(replies => {
                const respondents = Object.keys(replies);
                shuffle(respondents);
                for (let respondent of respondents) {
                    // TODO: response delay
                    this.messages.push({ user: names[respondent], text: replies[respondent], author: respondent });
                }
            });
            this.input = '';
        }
    }
});
