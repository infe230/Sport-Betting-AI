document.querySelectorAll('.predictionForm').forEach(form => {
    form.addEventListener('submit', function(e) {
        e.preventDefault();

        const modelType = e.target.querySelector('button').getAttribute('data-model-type');
        const homeSide = e.target.querySelector('[name="home_side"]').value;
        const awaySide = e.target.querySelector('[name="away_side"]').value;

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'model_type': modelType,
                'home_side': homeSide,
                'away_side': awaySide
            })
        })
        .then(response => response.json())
        .then(data => {
            const ctx = e.target.nextElementSibling.querySelector('.resultsChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Home Win Probability', 'Away Win Probability'],
                    datasets: [{
                        label: 'Prediction',
                        data: [data.predictions.home_win_prob, data.predictions.away_win_prob],
                        backgroundColor: ['rgba(75, 192, 192, 0.2)', 'rgba(255, 99, 132, 0.2)'],
                        borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 99, 132, 1)'],
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        });
    });
});
