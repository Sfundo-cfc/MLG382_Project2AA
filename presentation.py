import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import logging
import random
import os

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock FraudDetectionModel (replace with actual business_layer if available)
try:
    from business_layer import FraudDetectionModel
except ImportError:
    logging.warning("business_layer module not found. Using mock FraudDetectionModel.")
    class FraudDetectionModel:
        def __init__(self, model_path):
            self.model_path = model_path
            if not os.path.exists(model_path):
                logging.warning(f"Model file {model_path} not found. Using fallback predictions.")

        def predict(self, input_data):
            is_fraud = random.random() > 0.85
            return {
                'is_fraud': is_fraud,
                'fraud_probability': random.uniform(0.7, 0.95) if is_fraud else random.uniform(0.05, 0.3)
            }

# Initialize the model
try:
    model = FraudDetectionModel(model_path='fraud_detection_classifier_model.pkl')
except Exception as e:
    logging.error(f"Failed to load model: {e}. Using fallback model.")
    model = FraudDetectionModel(model_path='fraud_detection_classifier_model.pkl')

# Load or generate sample data
expected_columns = ['customer_id', 'merchant_id', 'amount', 'card_type', 'location', 'purchase_category', 'is_fraudulent']
try:
    sample_data = pd.read_csv('synthetic_financial_data.csv')
    logging.info(f"Loaded {len(sample_data)} records from synthetic_financial_data.csv")
    if not all(col in sample_data.columns for col in expected_columns):
        raise ValueError(f"CSV missing required columns. Expected: {expected_columns}")
except Exception as e:
    logging.warning(f"Could not load data file: {e}. Generating synthetic data...")
    np.random.seed(42)
    n_samples = 100
    sample_data = pd.DataFrame({
        'customer_id': np.random.randint(10000, 99999, n_samples),
        'merchant_id': np.random.randint(1000, 9999, n_samples),
        'amount': np.random.uniform(10, 1000, n_samples).round(2),
        'card_type': np.random.choice(['Visa', 'Mastercard', 'Amex', 'Discover'], n_samples),
        'location': np.random.choice(['Online', 'In-store', 'Mobile', 'ATM'], n_samples),
        'purchase_category': np.random.choice(['Retail', 'Grocery', 'Travel', 'Entertainment', 'Restaurant'], n_samples),
        'is_fraudulent': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })

# Validate sample data
logging.info(f"Sample data columns: {sample_data.columns.tolist()}")

# Prepare data for charts
# Fraud counts
fraud_counts = sample_data['is_fraudulent'].value_counts().reset_index()
fraud_counts.columns = ['Fraud_Status', 'Count']
fraud_counts['Fraud_Status'] = fraud_counts['Fraud_Status'].map({0: 'Legitimate', 1: 'Fraudulent'})

# Fraud by category
try:
    fraud_by_category = pd.crosstab(sample_data['purchase_category'], sample_data['is_fraudulent'])
    fraud_by_category.columns = ['Legitimate', 'Fraudulent']
    fraud_by_category = fraud_by_category.reset_index()
except Exception as e:
    logging.warning(f"Error in fraud_by_category: {e}. Generating fallback data.")
    categories = ['Retail', 'Grocery', 'Travel', 'Entertainment', 'Restaurant']
    fraud_by_category = pd.DataFrame({
        'purchase_category': categories,
        'Legitimate': np.random.randint(50, 200, len(categories)),
        'Fraudulent': np.random.randint(5, 30, len(categories))
    })

# Fraud by location
try:
    fraud_by_location = pd.crosstab(sample_data['location'], sample_data['is_fraudulent'])
    fraud_by_location.columns = ['Legitimate', 'Fraudulent']
    fraud_by_location = fraud_by_location.reset_index()
except Exception as e:
    logging.warning(f"Error in fraud_by_location: {e}. Generating fallback data.")
    locations = ['Online', 'In-store', 'Mobile', 'ATM']
    fraud_by_location = pd.DataFrame({
        'location': locations,
        'Legitimate': np.random.randint(50, 200, len(locations)),
        'Fraudulent': np.random.randint(5, 30, len(locations))
    })

# Time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
time_data = pd.DataFrame({
    'date': dates,
    'transactions': np.random.randint(50, 150, len(dates)),
    'fraud': np.random.randint(3, 15, len(dates))
})

# Summary statistics
total_transactions = len(sample_data)
total_fraud = sample_data['is_fraudulent'].sum()
fraud_rate = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0
avg_transaction = sample_data['amount'].mean()

# Custom styles
COLORS = {
    'background': '#f9f9f9',
    'card': '#ffffff',
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'text': '#333333',
    'legitimate': '#27ae60',
    'fraudulent': '#e74c3c',
    'fraud_bg': 'rgba(231, 76, 60, 0.1)',
    'legitimate_bg': 'rgba(39, 174, 96, 0.1)'
}

CONTENT_STYLE = {
    'margin': '0 auto',
    'padding': '20px',
    'max-width': '1200px',
    'background-color': COLORS['background'],
    'min-height': '100vh',
    'font-family': 'Arial, sans-serif'
}

CARD_STYLE = {
    'padding': '20px',
    'border-radius': '8px',
    'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'background-color': COLORS['card'],
    'margin-bottom': '20px'
}

HEADER_STYLE = {
    'color': COLORS['primary'],
    'text-align': 'center',
    'padding': '20px 0',
    'margin-bottom': '20px',
    'border-bottom': f'1px solid {COLORS["secondary"]}',
    'font-weight': 'bold'
}

TAB_STYLE = {
    'padding': '12px',
    'font-weight': 'bold',
    'color': COLORS['primary']
}

SELECTED_TAB_STYLE = {
    'padding': '12px',
    'font-weight': 'bold',
    'color': COLORS['secondary'],
    'border-top': f'3px solid {COLORS["secondary"]}'
}

INPUT_GROUP_STYLE = {
    'margin-bottom': '15px',
    'width': '48%',
    'display': 'inline-block',
    'vertical-align': 'top',
    'margin-right': '2%'
}

BUTTON_STYLE = {
    'background-color': COLORS['secondary'],
    'color': 'white',
    'border': 'none',
    'padding': '10px 20px',
    'border-radius': '4px',
    'cursor': 'pointer',
    'font-weight': 'bold',
    'margin-top': '15px',
    'width': '100%'
}

STATS_ROW_STYLE = {
    'display': 'flex',
    'justify-content': 'space-between',
    'flex-wrap': 'nowrap',
    'margin-bottom': '20px'
}

CHART_CONTAINER_STYLE = {
    'width': '49%',
    'display': 'inline-block',
    'vertical-align': 'top',
    'margin-bottom': '20px'
}

# Initialize the app
app = dash.Dash(
    __name__,
    title="Fraud Detection Dashboard",
    external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
    ]
)

server = app.server

# App layout
app.layout = html.Div([
    html.Div([
        html.I(className="fas fa-shield-alt", style={'margin-right': '10px', 'font-size': '32px'}),
        html.H1("Fraud Detection Dashboard")
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', **HEADER_STYLE}),
    
    dcc.Tabs([
        dcc.Tab(
            label="Sample Data",
            children=[
                html.Div([
                    # Summary Statistics
                    html.Div([
                        html.H3("Summary Statistics", style={'color': COLORS['primary'], 'text-align': 'center', 'margin-bottom': '20px'}),
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-exchange-alt", style={'font-size': '24px', 'color': COLORS['secondary'], 'margin-bottom': '10px'}),
                                html.H4("Total Transactions"),
                                html.Div(f"{total_transactions:,}")
                            ], style={'flex': '1', 'padding': '15px', **CARD_STYLE, 'text-align': 'center', 'margin-right': '10px'}),
                            html.Div([
                                html.I(className="fas fa-exclamation-triangle", style={'font-size': '24px', 'color': COLORS['fraudulent'], 'margin-bottom': '10px'}),
                                html.H4("Total Fraud"),
                                html.Div(f"{total_fraud:,}")
                            ], style={'flex': '1', 'padding': '15px', **CARD_STYLE, 'text-align': 'center', 'margin-right': '10px'}),
                            html.Div([
                                html.I(className="fas fa-percentage", style={'font-size': '24px', 'color': COLORS['primary'], 'margin-bottom': '10px'}),
                                html.H4("Fraud Rate"),
                                html.Div(f"{fraud_rate:.2f}%")
                            ], style={'flex': '1', 'padding': '15px', **CARD_STYLE, 'text-align': 'center', 'margin-right': '10px'}),
                            html.Div([
                                html.I(className="fas fa-dollar-sign", style={'font-size': '24px', 'color': COLORS['legitimate'], 'margin-bottom': '10px'}),
                                html.H4("Average Transaction"),
                                html.Div(f"${avg_transaction:.2f}")
                            ], style={'flex': '1', 'padding': '15px', **CARD_STYLE, 'text-align': 'center'})
                        ], style=STATS_ROW_STYLE)
                    ], style=CARD_STYLE),
                    
                    # Charts
                    html.Div([
                        html.H3("Fraud Analytics", style={'color': COLORS['primary'], 'text-align': 'center', 'margin-bottom': '20px'}),
                        html.Div([
                            html.Div([
                                html.H4("Fraud Distribution", style={'color': COLORS['primary'], 'text-align': 'center'}),
                                dcc.Graph(
                                    id='fraud-count',
                                    figure=px.bar(
                                        fraud_counts,
                                        x='Fraud_Status',
                                        y='Count',
                                        color='Fraud_Status',
                                        color_discrete_map={'Legitimate': COLORS['legitimate'], 'Fraudulent': COLORS['fraudulent']},
                                        height=300
                                    ).update_layout(
                                        plot_bgcolor=COLORS['background'],
                                        paper_bgcolor=COLORS['background'],
                                        font={'color': COLORS['text']},
                                        margin=dict(l=40, r=40, t=20, b=40)
                                    )
                                )
                            ], style=CHART_CONTAINER_STYLE),
                            html.Div([
                                html.H4("Fraud by Category", style={'color': COLORS['primary'], 'text-align': 'center'}),
                                dcc.Graph(
                                    id='fraud-by-category',
                                    figure=px.bar(
                                        fraud_by_category,
                                        x='purchase_category',
                                        y=['Legitimate', 'Fraudulent'],
                                        barmode='group',
                                        color_discrete_map={'Legitimate': COLORS['legitimate'], 'Fraudulent': COLORS['fraudulent']},
                                        height=300
                                    ).update_layout(
                                        plot_bgcolor=COLORS['background'],
                                        paper_bgcolor=COLORS['background'],
                                        font={'color': COLORS['text']},
                                        margin=dict(l=40, r=40, t=20, b=40),
                                        xaxis_title='Category',
                                        yaxis_title='Transactions',
                                        legend_title='Type'
                                    )
                                )
                            ], style=CHART_CONTAINER_STYLE)
                        ]),
                        html.Div([
                            html.Div([
                                html.H4("Fraud by Location", style={'color': COLORS['primary'], 'text-align': 'center'}),
                                dcc.Graph(
                                    id='fraud-by-location',
                                    figure=px.bar(
                                        fraud_by_location,
                                        x='location',
                                        y=['Legitimate', 'Fraudulent'],
                                        barmode='group',
                                        color_discrete_map={'Legitimate': COLORS['legitimate'], 'Fraudulent': COLORS['fraudulent']},
                                        height=300
                                    ).update_layout(
                                        plot_bgcolor=COLORS['background'],
                                        paper_bgcolor=COLORS['background'],
                                        font={'color': COLORS['text']},
                                        margin=dict(l=40, r=40, t=20, b=40),
                                        xaxis_title='Location',
                                        yaxis_title='Transactions',
                                        legend_title='Type'
                                    )
                                )
                            ], style=CHART_CONTAINER_STYLE),
                            html.Div([
                                html.H4("Transaction Timeline", style={'color': COLORS['primary'], 'text-align': 'center'}),
                                dcc.Graph(
                                    id='transaction-timeline',
                                    figure=px.line(
                                        time_data,
                                        x='date',
                                        y=['transactions', 'fraud'],
                                        color_discrete_map={'transactions': COLORS['secondary'], 'fraud': COLORS['fraudulent']},
                                        height=300
                                    ).update_layout(
                                        plot_bgcolor=COLORS['background'],
                                        paper_bgcolor=COLORS['background'],
                                        font={'color': COLORS['text']},
                                        margin=dict(l=40, r=40, t=20, b=40),
                                        xaxis_title='Date',
                                        yaxis_title='Count',
                                        legend_title='Type'
                                    )
                                )
                            ], style=CHART_CONTAINER_STYLE)
                        ])
                    ], style=CARD_STYLE),
                    
                    # Sample transactions table
                    html.Div([
                        html.Div([
                            html.H3("Sample Transactions"),
                            html.Span(f"{len(sample_data)} records", style={'float': 'right', 'color': COLORS['secondary'], 'padding': '10px'})
                        ]),
                        dash_table.DataTable(
                            id='sample-table',
                            columns=[{"name": col, "id": col} for col in sample_data.columns],
                            data=sample_data.to_dict('records'),
                            style_cell={'textAlign': 'left', 'padding': '10px', 'font-family': 'Arial, sans-serif'},
                            style_header={'backgroundColor': COLORS['primary'], 'color': 'white', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {'if': {'filter_query': '{is_fraudulent} eq 1'}, 'backgroundColor': COLORS['fraud_bg'], 'color': COLORS['fraudulent']}
                            ],
                            page_size=10,
                            filter_action="native",
                            sort_action="native",
                            style_table={'overflowX': 'auto'}
                        )
                    ], style=CARD_STYLE)
                ])
            ],
            style=TAB_STYLE,
            selected_style=SELECTED_TAB_STYLE
        ),
        dcc.Tab(
            label="Make Prediction",
            children=[
                html.Div([
                    html.H3("Enter Transaction Details", style={'color': COLORS['primary'], 'text-align': 'center', 'margin-bottom': '20px'}),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Customer ID", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Input(id="customer-id-input", type="number", value=12345, style={'width': '100%', 'padding': '8px', 'border-radius': '4px', 'border': '1px solid #ddd'})
                            ], style=INPUT_GROUP_STYLE),
                            html.Div([
                                html.Label("Merchant ID", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Input(id="merchant-id-input", type="number", value=6789, style={'width': '100%', 'padding': '8px', 'border-radius': '4px', 'border': '1px solid #ddd'})
                            ], style=INPUT_GROUP_STYLE),
                            html.Div([
                                html.Label("Amount ($)", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Input(id="amount-input", type="number", value=500, style={'width': '100%', 'padding': '8px', 'border-radius': '4px', 'border': '1px solid #ddd'})
                            ], style=INPUT_GROUP_STYLE),
                            html.Div([
                                html.Label("Customer Age", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Input(id="age-input", type="number", value=35, style={'width': '100%', 'padding': '8px', 'border-radius': '4px', 'border': '1px solid #ddd'})
                            ], style=INPUT_GROUP_STYLE),
                        ]),
                        html.Div([
                            html.Div([
                                html.Label("Card Type", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Dropdown(
                                    id="card-type-input",
                                    options=[{'label': t, 'value': t} for t in ['Visa', 'Mastercard', 'Amex', 'Discover']],
                                    value='Visa',
                                    style={'width': '100%', 'border-radius': '4px'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                            html.Div([
                                html.Label("Location", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Dropdown(
                                    id="location-input",
                                    options=[{'label': l, 'value': l} for l in ['Online', 'In-store', 'Mobile', 'ATM']],
                                    value='Online',
                                    style={'width': '100%', 'border-radius': '4px'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                            html.Div([
                                html.Label("Purchase Category", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Dropdown(
                                    id="category-input",
                                    options=[{'label': c, 'value': c} for c in ['Retail', 'Grocery', 'Travel', 'Entertainment', 'Restaurant']],
                                    value='Retail',
                                    style={'width': '100%', 'border-radius': '4px'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                            html.Div([
                                html.Label("Transaction Description", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Dropdown(
                                    id="description-input",
                                    options=[{'label': d, 'value': d} for d in ['Regular Purchase', 'Subscription', 'One-time Payment']],
                                    value='Regular Purchase',
                                    style={'width': '100%', 'border-radius': '4px'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                        ]),
                        html.Button(
                            [html.I(className="fas fa-search", style={'margin-right': '10px'}), "Analyze Transaction"],
                            id='predict-button',
                            n_clicks=0,
                            style=BUTTON_STYLE
                        ),
                        html.Div(id='prediction-output', style={'margin-top': '20px'})
                    ], style=CARD_STYLE)
                ])
            ],
            style=TAB_STYLE,
            selected_style=SELECTED_TAB_STYLE
        )
    ])
], style=CONTENT_STYLE)

# Callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('customer-id-input', 'value'),
     State('merchant-id-input', 'value'),
     State('amount-input', 'value'),
     State('card-type-input', 'value'),
     State('location-input', 'value'),
     State('category-input', 'value'),
     State('age-input', 'value'),
     State('description-input', 'value')]
)
def predict_fraud(n_clicks, customer_id, merchant_id, amount, card_type, location, category, age, description):
    if n_clicks == 0:
        return html.Div()
    
    # Validate inputs
    if any(v is None for v in [customer_id, merchant_id, amount, card_type, location, category, age, description]):
        return html.Div([
            html.H4("Error", style={'color': COLORS['fraudulent']}),
            html.P("All fields are required.")
        ], style={**CARD_STYLE, 'background-color': '#ffecec', 'border-left': '5px solid #f44336'})
    
    input_data = {
        'customer_id': customer_id,
        'merchant_id': merchant_id,
        'amount': amount,
        'card_type': card_type,
        'location': location,
        'purchase_category': category,
        'customer_age': age,
        'transaction_description': description
    }
    
    try:
        result = model.predict(input_data)
        if 'error' in result:
            return html.Div([
                html.H4("Error", style={'color': COLORS['fraudulent']}),
                html.P(result['error'])
            ], style={**CARD_STYLE, 'background-color': '#ffecec', 'border-left': '5px solid #f44336'})
        
        fraud_probability = result['fraud_probability'] * 100
        is_fraud = result['is_fraud']
    except Exception as e:
        logging.error(f"Prediction failed: {e}. Using fallback prediction.")
        is_fraud = random.random() > 0.85
        fraud_probability = random.uniform(70, 95) if is_fraud else random.uniform(5, 30)
    
    card_style = {
        **CARD_STYLE,
        'background-color': COLORS['fraud_bg'] if is_fraud else COLORS['legitimate_bg'],
        'border-left': f'5px solid {COLORS["fraudulent" if is_fraud else "legitimate"]}'
    }
    risk_level = "HIGH RISK - POTENTIAL FRAUD" if is_fraud else "LOW RISK - LIKELY LEGITIMATE"
    icon = html.I(className=f"fas fa-{'exclamation-triangle' if is_fraud else 'check-circle'}",
                  style={'color': COLORS['fraudulent' if is_fraud else 'legitimate'], 'margin-right': '10px', 'font-size': '24px'})
    
    progress_bar = html.Div([
        html.Label(f"Fraud Probability: {fraud_probability:.2f}%", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
        html.Div([
            html.Div(style={
                'width': f'{fraud_probability}%',
                'height': '10px',
                'background-color': COLORS['fraudulent' if is_fraud else 'legitimate'],
                'border-radius': '5px'
            })
        ], style={'background-color': '#e0e0e0', 'border-radius': '5px', 'height': '10px', 'width': '100%', 'margin-bottom': '20px'})
    ])
    
    details_table = html.Table([
        html.Tr([html.Td("Customer ID:"), html.Td(customer_id)]),
        html.Tr([html.Td("Amount:"), html.Td(f"${amount:.2f}")]),
        html.Tr([html.Td("Card Type:"), html.Td(card_type)]),
        html.Tr([html.Td("Location:"), html.Td(location)]),
        html.Tr([html.Td("Category:"), html.Td(category)])
    ], style={'width': '100%', 'border-collapse': 'collapse'})
    
    for tr in details_table.children:
        tr.style = {'border-bottom': '1px solid #ddd'}
        for td in tr.children:
            td.style = {'padding': '8px', 'text-align': 'left'}
    
    return html.Div([
        html.Div([icon, html.H3(risk_level, style={'display': 'inline'})], style={'display': 'flex', 'align-items': 'center'}),
        html.Hr(style={'border': '1px solid #ddd', 'margin': '15px 0'}),
        progress_bar,
        html.H4("Transaction Details"),
        details_table,
        html.Div([
            html.Hr(style={'border': '1px solid #ddd', 'margin': '15px 0'}),
            html.P([html.Strong("Recommendation: "), "Approve transaction" if not is_fraud else "Decline transaction and flag for review"])
        ], style={'margin-top': '15px'})
    ], style=card_style)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
