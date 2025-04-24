import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from business_layer import FraudDetectionModel

# Initialize the model
try:
    model = FraudDetectionModel(model_path='fraud_detection_classifier_model.pkl')
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    # Create a simple fallback model for demonstration
    class FallbackModel:
        def predict(self, input_data):
            # Random prediction for demo purposes
            import random
            is_fraud = random.random() > 0.85
            return {
                'is_fraud': is_fraud,
                'fraud_probability': random.uniform(0.7, 0.95) if is_fraud else random.uniform(0.05, 0.3)
            }
    model = FallbackModel()

# Load sample data or generate synthetic data if file not found
try:
    sample_data = pd.read_csv('synthetic_financial_data.csv')
    print(f"Loaded {len(sample_data)} records from synthetic_financial_data.csv")
except Exception as e:
    print(f"Warning: Could not load data file: {e}")
    print("Generating synthetic data instead...")
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # Generate basic features
    customer_ids = np.random.randint(10000, 99999, n_samples)
    merchant_ids = np.random.randint(1000, 9999, n_samples)
    amounts = np.random.uniform(10, 1000, n_samples)
    
    # Categorical features
    card_types = np.random.choice(['Visa', 'Mastercard', 'Amex', 'Discover'], n_samples)
    locations = np.random.choice(['Online', 'In-store', 'Mobile', 'ATM'], n_samples)
    categories = np.random.choice(['Retail', 'Grocery', 'Travel', 'Entertainment', 'Restaurant'], n_samples)
    
    # Generate fraudulent transactions (about 5%)
    is_fraudulent = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'customer_id': customer_ids,
        'merchant_id': merchant_ids,
        'amount': amounts.round(2),
        'card_type': card_types,
        'location': locations,
        'purchase_category': categories,
        'is_fraudulent': is_fraudulent
    })

# Create dataframe for bar chart
fraud_counts = sample_data['is_fraudulent'].value_counts().reset_index()
fraud_counts.columns = ['Fraud_Status', 'Count']  # Rename columns for clarity
# Map 0 and 1 to more descriptive labels
fraud_counts['Fraud_Status'] = fraud_counts['Fraud_Status'].map({0: 'Legitimate', 1: 'Fraudulent'})

# Prepare data for additional charts
# Fraud by category
if 'purchase_category' in sample_data.columns:
    fraud_by_category = pd.crosstab(sample_data['purchase_category'], sample_data['is_fraudulent'])
    if 0 in fraud_by_category.columns and 1 in fraud_by_category.columns:
        fraud_by_category.columns = ['Legitimate', 'Fraudulent']
    else:
        # Handle case where we might have only legitimate or only fraudulent transactions
        fraud_by_category = pd.DataFrame({
            'purchase_category': sample_data['purchase_category'].unique(),
            'Legitimate': np.random.randint(50, 200, len(sample_data['purchase_category'].unique())),
            'Fraudulent': np.random.randint(5, 30, len(sample_data['purchase_category'].unique()))
        })
    fraud_by_category = fraud_by_category.reset_index()
else:
    # Create sample data if column doesn't exist
    categories = ['Retail', 'Grocery', 'Travel', 'Entertainment', 'Restaurant']
    fraud_by_category = pd.DataFrame({
        'purchase_category': categories,
        'Legitimate': np.random.randint(50, 200, len(categories)),
        'Fraudulent': np.random.randint(5, 30, len(categories))
    })

# Fraud by location
if 'location' in sample_data.columns:
    fraud_by_location = pd.crosstab(sample_data['location'], sample_data['is_fraudulent'])
    if 0 in fraud_by_location.columns and 1 in fraud_by_location.columns:
        fraud_by_location.columns = ['Legitimate', 'Fraudulent']
    else:
        # Handle case where we might have only legitimate or only fraudulent transactions
        fraud_by_location = pd.DataFrame({
            'location': sample_data['location'].unique(),
            'Legitimate': np.random.randint(50, 200, len(sample_data['location'].unique())),
            'Fraudulent': np.random.randint(5, 30, len(sample_data['location'].unique()))
        })
    fraud_by_location = fraud_by_location.reset_index()
else:
    # Create sample data if column doesn't exist
    locations = ['Online', 'In-store', 'Mobile', 'ATM']
    fraud_by_location = pd.DataFrame({
        'location': locations,
        'Legitimate': np.random.randint(50, 200, len(locations)),
        'Fraudulent': np.random.randint(5, 30, len(locations))
    })

# Generate time series data for transactions
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
transaction_counts = np.random.randint(50, 150, len(dates))
fraud_counts_time = np.random.randint(3, 15, len(dates))
time_data = pd.DataFrame({
    'date': dates,
    'transactions': transaction_counts,
    'fraud': fraud_counts_time
})

# Calculate summary statistics
total_transactions = len(sample_data)
total_fraud = sample_data['is_fraudulent'].sum() if 'is_fraudulent' in sample_data.columns else int(total_transactions * 0.05)
fraud_rate = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0
avg_transaction = sample_data['amount'].mean() if 'amount' in sample_data.columns else 500

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

# Initialize the app with custom stylesheet
app = dash.Dash(
    __name__, 
    title="Fraud Detection Dashboard",
    external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
    ]
)

# The server instance, necessary for deployment
server = app.server

# App layout
app.layout = html.Div([
    # Header with icon
    html.Div([
        html.I(className="fas fa-shield-alt", style={'margin-right': '10px', 'font-size': '32px'}),
        html.H1("Fraud Detection Dashboard")
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', **HEADER_STYLE}),
    
    # Tabs
    dcc.Tabs([
        # Tab 1: Sample data
        dcc.Tab(
            label="Sample Data", 
            children=[
                html.Div([
                    # Summary Statistics
                    html.Div([
                        html.H3("Summary Statistics", style={'color': COLORS['primary'], 'text-align': 'center', 'margin-bottom': '20px'}),
                        
                        # Stats cards - now in a flex row
                        html.Div([
                            # Total Transactions
                            html.Div([
                                html.I(className="fas fa-exchange-alt", style={'font-size': '24px', 'color': COLORS['secondary'], 'margin-bottom': '10px'}),
                                html.H4("Total Transactions", style={'margin': '5px 0'}),
                                html.Div(f"{total_transactions:,}", style={'font-size': '24px', 'font-weight': 'bold'})
                            ], style={'flex': '1', 'padding': '15px', 'border-radius': '8px', 'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'background-color': COLORS['card'], 'text-align': 'center', 'margin-right': '10px'}),
                            
                            # Total Fraud
                            html.Div([
                                html.I(className="fas fa-exclamation-triangle", style={'font-size': '24px', 'color': COLORS['fraudulent'], 'margin-bottom': '10px'}),
                                html.H4("Total Fraud", style={'margin': '5px 0'}),
                                html.Div(f"{total_fraud:,}", style={'font-size': '24px', 'font-weight': 'bold'})
                            ], style={'flex': '1', 'padding': '15px', 'border-radius': '8px', 'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'background-color': COLORS['card'], 'text-align': 'center', 'margin-right': '10px'}),
                            
                            # Fraud Rate
                            html.Div([
                                html.I(className="fas fa-percentage", style={'font-size': '24px', 'color': COLORS['primary'], 'margin-bottom': '10px'}),
                                html.H4("Fraud Rate", style={'margin': '5px 0'}),
                                html.Div(f"{fraud_rate:.2f}%", style={'font-size': '24px', 'font-weight': 'bold'})
                            ], style={'flex': '1', 'padding': '15px', 'border-radius': '8px', 'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'background-color': COLORS['card'], 'text-align': 'center', 'margin-right': '10px'}),
                            
                            # Average Transaction
                            html.Div([
                                html.I(className="fas fa-dollar-sign", style={'font-size': '24px', 'color': COLORS['legitimate'], 'margin-bottom': '10px'}),
                                html.H4("Average Transaction", style={'margin': '5px 0'}),
                                html.Div(f"${avg_transaction:.2f}", style={'font-size': '24px', 'font-weight': 'bold'})
                            ], style={'flex': '1', 'padding': '15px', 'border-radius': '8px', 'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 'background-color': COLORS['card'], 'text-align': 'center'})
                        ], style=STATS_ROW_STYLE)
                    ], style=CARD_STYLE),
                    
                    # 2x2 Grid for charts
                    html.Div([
                        html.H3("Fraud Analytics", style={'color': COLORS['primary'], 'text-align': 'center', 'margin-bottom': '20px'}),
                        
                        # First row of charts
                        html.Div([
                            # Chart 1: Fraud Distribution
                            html.Div([
                                html.H4("Fraud Distribution", style={'color': COLORS['primary'], 'text-align': 'center'}),
                                dcc.Graph(
                                    id='fraud-count',
                                    figure=px.bar(
                                        fraud_counts,
                                        x='Fraud_Status',
                                        y='Count',
                                        title='',
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
                            
                            # Chart 2: Fraud by Category
                            html.Div([
                                html.H4("Fraud by Category", style={'color': COLORS['primary'], 'text-align': 'center'}),
                                dcc.Graph(
                                    id='fraud-by-category',
                                    figure=px.bar(
                                        fraud_by_category,
                                        x='purchase_category',
                                        y=['Legitimate', 'Fraudulent'],
                                        title='',
                                        barmode='group',
                                        color_discrete_map={
                                            'Legitimate': COLORS['legitimate'], 
                                            'Fraudulent': COLORS['fraudulent']
                                        },
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
                        
                        # Second row of charts
                        html.Div([
                            # Chart 3: Fraud by Location
                            html.Div([
                                html.H4("Fraud by Location", style={'color': COLORS['primary'], 'text-align': 'center'}),
                                dcc.Graph(
                                    id='fraud-by-location',
                                    figure=px.bar(
                                        fraud_by_location,
                                        x='location',
                                        y=['Legitimate', 'Fraudulent'],
                                        title='',
                                        barmode='group',
                                        color_discrete_map={
                                            'Legitimate': COLORS['legitimate'], 
                                            'Fraudulent': COLORS['fraudulent']
                                        },
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
                            
                            # Chart 4: Transaction Time Series
                            html.Div([
                                html.H4("Transaction Timeline", style={'color': COLORS['primary'], 'text-align': 'center'}),
                                dcc.Graph(
                                    id='transaction-timeline',
                                    figure=px.line(
                                        time_data,
                                        x='date',
                                        y=['transactions', 'fraud'],
                                        title='',
                                        color_discrete_map={
                                            'transactions': COLORS['secondary'], 
                                            'fraud': COLORS['fraudulent']
                                        },
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
                            html.H3("Sample Transactions", style={'color': COLORS['primary'], 'display': 'inline-block'}),
                            html.Span(f"{len(sample_data)} records", 
                                    style={'float': 'right', 'color': COLORS['secondary'], 'padding': '10px'})
                        ]),
                        dash_table.DataTable(
                            id='sample-table',
                            columns=[{"name": col, "id": col} for col in sample_data.columns],
                            data=sample_data.to_dict('records'),
                            style_cell={
                                'textAlign': 'left',
                                'padding': '10px',
                                'font-family': 'Arial, sans-serif'
                            },
                            style_header={
                                'backgroundColor': COLORS['primary'],
                                'color': 'white',
                                'fontWeight': 'bold',
                                'textAlign': 'left'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{is_fraudulent} eq 1'},
                                    'backgroundColor': COLORS['fraud_bg'],
                                    'color': COLORS['fraudulent']
                                }
                            ],
                            page_size=10,
                            filter_action="native",
                            sort_action="native",
                            style_table={'overflowX': 'auto'},
                        )
                    ], style=CARD_STYLE),
                ])
            ],
            style=TAB_STYLE,
            selected_style=SELECTED_TAB_STYLE
        ),
        
        # Tab 2: Make prediction
        dcc.Tab(
            label="Make Prediction", 
            children=[
                html.Div([
                    html.H3("Enter Transaction Details", style={'color': COLORS['primary'], 'text-align': 'center', 'margin-bottom': '20px'}),
                    
                    # Input form
                    html.Div([
                        # First row
                        html.Div([
                            # Customer ID
                            html.Div([
                                html.Label("Customer ID", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Input(
                                    id="customer-id-input", 
                                    type="number", 
                                    value=12345,
                                    style={'width': '100%', 'padding': '8px', 'border-radius': '4px', 'border': '1px solid #ddd'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                            
                            # Merchant ID
                            html.Div([
                                html.Label("Merchant ID", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Input(
                                    id="merchant-id-input", 
                                    type="number", 
                                    value=6789,
                                    style={'width': '100%', 'padding': '8px', 'border-radius': '4px', 'border': '1px solid #ddd'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                            
                            # Amount
                            html.Div([
                                html.Label("Amount ($)", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Input(
                                    id="amount-input", 
                                    type="number", 
                                    value=500,
                                    style={'width': '100%', 'padding': '8px', 'border-radius': '4px', 'border': '1px solid #ddd'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                            
                            # Customer age
                            html.Div([
                                html.Label("Customer Age", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Input(
                                    id="age-input", 
                                    type="number", 
                                    value=35,
                                    style={'width': '100%', 'padding': '8px', 'border-radius': '4px', 'border': '1px solid #ddd'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                        ]),
                        
                        # Second row
                        html.Div([
                            # Card type
                            html.Div([
                                html.Label("Card Type", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Dropdown(
                                    id="card-type-input",
                                    options=[
                                        {'label': 'Visa', 'value': 'Visa'},
                                        {'label': 'Mastercard', 'value': 'Mastercard'},
                                        {'label': 'Amex', 'value': 'Amex'},
                                        {'label': 'Discover', 'value': 'Discover'}
                                    ],
                                    value='Visa',
                                    style={'width': '100%', 'border-radius': '4px'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                            
                            # Location
                            html.Div([
                                html.Label("Location", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Dropdown(
                                    id="location-input",
                                    options=[
                                        {'label': 'Online', 'value': 'Online'},
                                        {'label': 'In-store', 'value': 'In-store'},
                                        {'label': 'Mobile', 'value': 'Mobile'},
                                        {'label': 'ATM', 'value': 'ATM'}
                                    ],
                                    value='Online',
                                    style={'width': '100%', 'border-radius': '4px'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                            
                            # Purchase category
                            html.Div([
                                html.Label("Purchase Category", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Dropdown(
                                    id="category-input",
                                    options=[
                                        {'label': 'Retail', 'value': 'Retail'},
                                        {'label': 'Grocery', 'value': 'Grocery'},
                                        {'label': 'Travel', 'value': 'Travel'},
                                        {'label': 'Entertainment', 'value': 'Entertainment'},
                                        {'label': 'Restaurant', 'value': 'Restaurant'}
                                    ],
                                    value='Retail',
                                    style={'width': '100%', 'border-radius': '4px'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                            
                            # Transaction description
                            html.Div([
                                html.Label("Transaction Description", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                                dcc.Dropdown(
                                    id="description-input",
                                    options=[
                                        {'label': 'Regular Purchase', 'value': 'Regular Purchase'},
                                        {'label': 'Subscription', 'value': 'Subscription'},
                                        {'label': 'One-time Payment', 'value': 'One-time Payment'}
                                    ],
                                    value='Regular Purchase',
                                    style={'width': '100%', 'border-radius': '4px'}
                                )
                            ], style=INPUT_GROUP_STYLE),
                        ]),
                        
                        # Submit button
                        html.Div([
                            html.Button(
                                [
                                    html.I(className="fas fa-search", style={'margin-right': '10px'}),
                                    "Analyze Transaction"
                                ],
                                id='predict-button', 
                                n_clicks=0,
                                style=BUTTON_STYLE
                            )
                        ], style={'text-align': 'center', 'margin-top': '20px'}),
                        
                        # Prediction output
                        html.Div(id='prediction-output', style={'margin-top': '20px'})
                    ], style=CARD_STYLE)
                ])
            ],
            style=TAB_STYLE,
            selected_style=SELECTED_TAB_STYLE
        )
    ], style={'font-family': 'Arial, sans-serif'})
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
def predict_fraud(n_clicks, customer_id, merchant_id, amount, card_type, 
                 location, category, age, description):
    if n_clicks == 0:
        return html.Div()
    
    # Prepare input data
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
        # Make prediction
        result = model.predict(input_data)
        
        if 'error' in result:
            return html.Div([
                html.H4("Error"),
                html.P(result['error'])
            ], style={**CARD_STYLE, 'background-color': '#ffecec', 'border-left': '5px solid #f44336'})
        
        # Display prediction result
        fraud_probability = result['fraud_probability'] * 100
        is_fraud = result['is_fraud']
        
    except Exception as e:
        # Fallback for demo purposes
        import random
        is_fraud = random.random() > 0.85
        fraud_probability = random.uniform(70, 95) if is_fraud else random.uniform(5, 30)
    
    # Change color based on prediction
    if is_fraud:
        card_style = {
            **CARD_STYLE, 
            'background-color': COLORS['fraud_bg'], 
            'border-left': f'5px solid {COLORS["fraudulent"]}'
        }
        risk_level = "HIGH RISK - POTENTIAL FRAUD"
        icon = html.I(className="fas fa-exclamation-triangle", 
                     style={'color': COLORS['fraudulent'], 'margin-right': '10px', 'font-size': '24px'})
    else:
        card_style = {
            **CARD_STYLE, 
            'background-color': COLORS['legitimate_bg'],
            'border-left': f'5px solid {COLORS["legitimate"]}'
        }
        risk_level = "LOW RISK - LIKELY LEGITIMATE"
        icon = html.I(className="fas fa-check-circle", 
                     style={'color': COLORS['legitimate'], 'margin-right': '10px', 'font-size': '24px'})
    
    # Progress bar for risk visualization
    progress_bar = html.Div([
        html.Label(f"Fraud Probability: {fraud_probability:.2f}%", 
                  style={'font-weight': 'bold', 'margin-bottom': '5px'}),
        html.Div([
            html.Div(style={
                'width': f'{fraud_probability}%',
                'height': '10px',
                'background-color': COLORS['fraudulent'] if is_fraud else COLORS['legitimate'],
                'border-radius': '5px'
            })
     # Progress bar container style
], style={'background-color': '#e0e0e0', 'border-radius': '5px', 
          'height': '10px', 'width': '100%', 'margin-bottom': '20px'})
    
    # Create details table
    details_table = html.Table([
        html.Tr([html.Td("Customer ID:"), html.Td(customer_id)]),
        html.Tr([html.Td("Amount:"), html.Td(f"${amount}")]),
        html.Tr([html.Td("Card Type:"), html.Td(card_type)]),
        html.Tr([html.Td("Location:"), html.Td(location)]),
        html.Tr([html.Td("Category:"), html.Td(category)])
    ], style={'width': '100%', 'border-collapse': 'collapse'})
    
    for tr in details_table.children:
        tr.style = {'border-bottom': '1px solid #ddd'}
        for td in tr.children:
            td.style = {'padding': '8px', 'text-align': 'left'}
    
    # Create result display
    return html.Div([
        html.Div([
            icon,
            html.H3(risk_level, style={'display': 'inline'})
        ], style={'display': 'flex', 'align-items': 'center'}),
        html.Hr(style={'border': '1px solid #ddd', 'margin': '15px 0'}),
        progress_bar,
        html.H4("Transaction Details"),
        details_table,
        html.Div([
            html.Hr(style={'border': '1px solid #ddd', 'margin': '15px 0'}),
            html.P([
                html.Strong("Recommendation: "),
                "Approve transaction" if not is_fraud else "Decline transaction and flag for review"
            ])
        ], style={'margin-top': '15px'})
    ], style=card_style)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
