import dash
from dash import dcc, html
from datetime import datetime as dt
from datetime import date
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import plotly.express as px
from model import train_lstm_model, make_predictions

# Instantiate Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'styles.css'])

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            [
                html.H1("Welcome to the Stock Dash App!", className="title"),
                html.Div([
                    html.Label("Enter Ticker Symbol: "),
                    dcc.Input(
                        id='stock-input',
                        type='text',
                        value='',
                        style={'width': '200px'}
                    ),
                    html.Button('Submit', id='submit-button', n_clicks=0),
                    html.Label("Select Date and Visualize Data: "),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=dt(2024, 3, 1),
                        end_date=dt.now().date(),  # Set end_date to current date
                        display_format='YYYY-MM-DD',
                        max_date_allowed=dt.now().date()  # Set maximum date allowed to current date
                    ),
                    html.Button('Stock Price', id='stock-price-button', n_clicks=0, style={'margin-top': '10px'}),
                    html.Button('Indicators', id='indicator-button', n_clicks=0, style={'margin-top': '10px'}),
                    html.Label("Enter number of days to forecast: "),
                    dcc.Input(
                        id='forecast-input',
                        type='text',
                        value='',
                        style={'width': '200px'}
                    ),
                    html.Button('Forecast', id='forecast-button', n_clicks=0),
                ]),
            ]
        ),
    ],
        class_name="row-1"),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(id='company-logo', src='', style={'height': '100px', 'width': 'auto'}),
                html.H2(id='company-name'),
                html.P(id='company-description', className='company-description'),
            ]),
        ]),
    ],
        class_name="row-2"),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([], id="stock-graph", className="graphs-content"),
                html.Div([], id="indicator-graph", className="graphs-content"),
                html.Div([], id="prediction-graph", className="graphs-content"),
            ]),
        ]),
    ],
        class_name="row-3"),
], fluid=True,
        class_name="main")

# Define callback functions
@app.callback(
    [dash.dependencies.Output('stock-graph', 'children')],
    [dash.dependencies.Input('stock-price-button', 'n_clicks')],
    [dash.dependencies.State('stock-input', 'value'), dash.dependencies.State('date-range', 'start_date'),
     dash.dependencies.State('date-range', 'end_date')]
)
def update_stock_graph(n_clicks, ticker, start_date, end_date):
    if n_clicks > 0 and ticker:
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty and not isinstance(df.index, pd.RangeIndex):
            df.reset_index(inplace=True)  # Reset the index
            fig = px.line(df, x='Date', y=['Open', 'Close'], title="Closing and Opening Price vs Date")
            return [dcc.Graph(figure=fig)]
        else:
            return ['No data available for the selected date range.']
    else:
        return ['']

@app.callback(
    [dash.dependencies.Output('indicator-graph', 'children')],
    [dash.dependencies.Input('indicator-button', 'n_clicks')],
    [dash.dependencies.State('stock-input', 'value'), dash.dependencies.State('date-range', 'start_date'),
     dash.dependencies.State('date-range', 'end_date')]
)

def update_indicator_graph(n_clicks, ticker, start_date, end_date):
    if n_clicks > 0 and ticker:
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty and not isinstance(df.index, pd.RangeIndex):
            df.reset_index(inplace=True)  # Reset the index
            df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            fig = px.scatter(df, x='Date', y='EWA_20', title="Exponential Moving Average vs Date")
            fig.update_traces(mode='lines')
            return [dcc.Graph(figure=fig)]
        else:
            return ['No data available for the selected date range.']
    else:
        return ['']


@app.callback(
    [dash.dependencies.Output('company-logo', 'src'),
     dash.dependencies.Output('company-name', 'children'),
     dash.dependencies.Output('company-description', 'children')],
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('stock-input', 'value')]
)

def update_logo_and_description(n_clicks, ticker):
    if n_clicks > 0 and ticker:
        try:
            company = yf.Ticker(ticker)
            logo_url = company.info.get('logo_url', '')
            description = company.info.get('longBusinessSummary', '')
            company_name = company.info.get('shortName', 'Company Name')
            print(f"Logo URL: {logo_url}")
            return logo_url, company_name, description
        except Exception as e:
            print(f"Error fetching data: {e}")
            return '', 'Failed to fetch company information', ''
    else:
        return '', '', ''

@app.callback(
    [dash.dependencies.Output('prediction-graph', 'children')],
    [dash.dependencies.Input('forecast-button', 'n_clicks')],
    [dash.dependencies.State('stock-input', 'value'), dash.dependencies.State('forecast-input', 'value')]
)
def update_prediction_graph(n_clicks, ticker, forecast_days):
    if n_clicks > 0 and ticker and forecast_days:
        try:
            df = yf.download(ticker, period="90d")
            model, scaler = train_lstm_model(df)
            predictions = make_predictions(model, scaler, df, int(forecast_days))
            fig = px.line(predictions, x='Date', y='Predicted Close', title="Predicted Closing Price vs Date")
            return [dcc.Graph(figure=fig)]
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return ['Error generating predictions.']
    else:
        return ['']

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
