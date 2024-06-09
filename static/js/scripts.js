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
            homeWinProb = data.predictions.home_win_prob;
            awayWinProb = data.predictions.away_win_prob;
            let message = "";

            if (homeWinProb > .80 || awayWinProb > .80) {
                message = "Bet ur life on this one.";
            } else if (homeWinProb > .70 || awayWinProb > .70) {
                message = "We definitely cashing out this parlay.";
            } else if (homeWinProb > .60 || awayWinProb > .60) {
                message = "Yeah I mean you could take this bet but ion know for sure just letting you know.";
            } else if (homeWinProb > .50 || awayWinProb > .50) {
                message = "Lowkey idk if this is the bet to make you rich.";
            } else {
                message = "Highkey I got no idea you should probably go bet on sumn else";
            }

            const ctx = e.target.nextElementSibling.querySelector('.resultsChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Home Win Probability', 'Draw Probability', 'Away Win Probability'],
                    datasets: [{
                        label: 'Prediction',
                        data: [data.predictions.home_win_prob, data.predictions.draw_prob, data.predictions.away_win_prob],
                        backgroundColor: ['rgba(75, 192, 192, 0.2)', 'rgba(255, 99, 132, 0.2)'],
                        borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 99, 132, 1)'],
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                font: {
                                    size: 16
                                }
                            }
                        },
                        x: {
                            ticks: {
                                font: {
                                    size: 16
                                }
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                font: {
                                    size: 18
                                }
                            }
                        },
                        tooltip: {
                            bodyFont: {
                                size: 16
                            },
                            titleFont: {
                                size: 16
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Prediction Results',
                        font: {
                            size: 20
                        }
                    }
                }
            });
            const messageContainer = document.createElement('div');
            messageContainer.textContent = message;
            messageContainer.style.fontSize = '18px';
            e.target.nextElementSibling.appendChild(messageContainer);
        });
    });
});
