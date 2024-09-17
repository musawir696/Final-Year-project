document.getElementById('detect-btn').addEventListener('click', async () => {
    const question = document.getElementById('question').value;
    const answer = document.getElementById('answer').value;

    const response = await fetch('/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question, answer })
    });

    const result = await response.json();
    document.getElementById('result').innerText = result.hallucinated ? 'Hallucinated' : 'Not Hallucinated';
});
