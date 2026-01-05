// static/js/charts.js
class TradingCharts {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.charts = {};
    }

    async renderCandlestick(data, layout = {}) {
        const plotData = [{
            x: data.map(d => d.timestamp),
            open: data.map(d => d.open),
            high: data.map(d => d.high),
            low: data.map(d => d.low),
            close: data.map(d => d.close),
            type: 'candlestick',
            name: 'Price',
            increasing: { line: { color: '#26a69a' } },
            decreasing: { line: { color: '#ef5350' } }
        }];

        const defaultLayout = {
            title: 'Price Chart',
            xaxis: { 
                rangeslider: { visible: false },
                type: 'date'
            },
            yaxis: { title: 'Price' },
            showlegend: false,
            margin: { t: 30, b: 30, l: 50, r: 30, pad: 0 }
        };

        const finalLayout = { ...defaultLayout, ...layout };
        Plotly.newPlot(this.container, plotData, finalLayout, { responsive: true });
    }

    addIndicator(data, name, color = '#2196F3') {
        const trace = {
            x: data.map(d => d.timestamp),
            y: data.map(d => d.value),
            type: 'scatter',
            mode: 'lines',
            name: name,
            line: { color: color }
        };

        Plotly.addTraces(this.container, [trace]);
    }

    updateChart(newData) {
        // Update chart with new data
        const update = {
            'y': [newData.close],
            'x': [[newData.timestamp]]
        };
        Plotly.extendTraces(this.container, update, [0]);
    }
}