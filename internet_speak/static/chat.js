function sendMessage(personality, message) {
    fetch(`/message/${encodeURIComponent(personality)}?message=${encodeURIComponent(message)}`)
        .then(response => {
            if (response.status >= 400) {
                let err = new Error(`HTTP ${response.status} error sending "${message} to ${personality}`);
                err.response = response;
                throw err;
            } else {
                return response.reply;
            }
        });
}