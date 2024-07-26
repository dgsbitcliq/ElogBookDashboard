import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
from sqlalchemy import create_engine,text
import urllib.parse
from dash_bootstrap_templates import ThemeSwitchAIO
from datetime import datetime, timedelta
import decimal
import seaborn
import plotly.express as px
import numpy as np
from dash_extensions import EventListener
from dash_extensions.enrich import DashProxy, html
import os

# Database connection setup
def get_database_connection():
    username = 'sa'
    password = urllib.parse.quote('BitDev#2017') 
    database_name = 'CLOUD_ELOGBOOK_PROD'
    server = '93.104.208.249,42223'
    connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(connection_string)
    return engine

# Fetch data from the database
def fetch_data(query):
    engine = get_database_connection()
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    return df

query = """
        SELECT DISTINCT
        VesselName, 
        TripNumber, 
        DepartureDate, 
        ArrivalDate, 
        FishingSetNumber, 
        FishSpecieName, 
        FishSize, 
        SUM(SUM(TotalWeight)) OVER (PARTITION BY FishSpecieName, FishSize, FishingSetID) AS TotalWeightPerSize, 
        SUM(SUM(TotalWeight)) OVER (PARTITION BY FishSpecieName, FishingSetID) AS TotalWeightPerSpecies, 
        SUM(SUM(TotalWeight)) OVER (PARTITION BY FishingSetID) AS TotalWeightPerSet, 
        SUM(SUM(TotalWeight)) OVER (PARTITION BY TripNumber) AS TotalWeightPerTrip ,
        Port,
        Pier, 
        SUM(SUM(UnloadingWeight)) OVER (PARTITION BY TripNumber) AS TotalUnloadingWeight,
        ROUND(CASE WHEN SUM(SUM(UnloadingWeight)) OVER (PARTITION BY TripNumber) = 0 THEN 0 ELSE SUM(SUM(TotalWeight)) OVER (PARTITION BY TripNumber) / SUM(SUM(UnloadingWeight)) OVER (PARTITION BY TripNumber) END, 2) AS 'P/D'
    FROM ( 
        SELECT 
            v.Name AS VesselName, 
            t.TripNumber, 
            t.DateStart AS DepartureDate, 
            t.DateEnd AS ArrivalDate, 
            fs.Number AS FishingSetNumber, 
            s.Name AS FishSpecieName, 
            fsize.Name AS FishSize, 
            fc.Weight AS TotalWeight, 
            fs.ID AS FishingSetID,
            p.Name AS Port, 
            pier.Name AS Pier, 
            ud.Weight AS UnloadingWeight 
        FROM 
            Vessel v 
            JOIN Trip t ON v.ID = t.VesselID 
            JOIN FishingZone fz ON t.TripNumber = fz.TripNumber AND v.ID = fz.VesselID 
            JOIN FishingSet fs ON fs.FishingZoneID = fz.ID 
            JOIN FishCatch fc ON fc.FishingSetID = fs.ID 
            JOIN FishSpecie s ON s.ID = fc.FishSpecieID 
            JOIN FishSize fsize ON fsize.ID = fc.FishSizeID 
            LEFT JOIN Unloading u ON t.TripNumber = u.TripNumber AND v.ID = u.VesselID 
            LEFT JOIN UnloadingData ud ON u.ID = ud.UnloadingID 
            LEFT JOIN Port p on u.PortID = p.ID 
            LEFT JOIN Pier pier on u.PierID = pier.ID 
    ) AS subquery 
    GROUP BY 
        VesselName, 
        TripNumber, 
        DepartureDate, 
        ArrivalDate, 
        FishingSetNumber, 
        FishSpecieName, 
        FishSize, 
        FishingSetID ,
        Port,
        Pier,
        UnloadingWeight
    ORDER BY 
        VesselName, 
        TripNumber, 
        FishingSetNumber, 
        FishSpecieName, 
        FishSize
        """
df = fetch_data(query)

decimal_cols = ['TotalWeightPerTrip', 'TotalWeightPerSize', 'TotalWeightPerSpecies', 'TotalWeightPerSet']
for col in decimal_cols:
    df[col] = df[col].apply(lambda x: float(x) if isinstance(x, decimal.Decimal) else x)

# Initialize Dash app
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
app.config.suppress_callback_exceptions = True  # Suppress callback exceptions

theme_switch = ThemeSwitchAIO(
    aio_id="theme", themes=[dbc.themes.CERULEAN, dbc.themes.DARKLY]
)

engine = get_database_connection()

def format_repeating_values(df):
    formatted_df = df.copy()
    vessel_trip_change_mask = (
        (formatted_df['VesselName'] != formatted_df['VesselName'].shift(1)) |
        (formatted_df['TripNumber'] != formatted_df['TripNumber'].shift(1))
    )
    for col in formatted_df.columns:
        if pd.api.types.is_datetime64_any_dtype(formatted_df[col]):
            formatted_df[col] = formatted_df[col].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else '')

        previous_values = formatted_df[col].shift(1)
        formatted_df[col] = formatted_df.apply(
            lambda row: row[col] if (row[col] != previous_values[row.name] or vessel_trip_change_mask[row.name])
            else '',
            axis=1
        )
    
    return formatted_df


# Define the layout
app.layout = dbc.Container([
    theme_switch,
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
], fluid=True)


# Define the callbacks
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):     

    df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])
    df['ArrivalDate'] = pd.to_datetime(df['ArrivalDate'])

    query_vessel_weights ="""select v.Name as VesselName,t.VesselID,t.TripNumber,fc.Weight, t.DateStart as DepartureDate, t.DateEnd AS ArrivalDate from Vessel v join Trip t on v.Id = t.VesselID join FishingZone fz ON t.TripNumber = fz.TripNumber AND v.ID = fz.VesselID JOIN FishingSet fs ON fs.FishingZoneID = fz.ID JOIN FishCatch fc ON fc.FishingSetID = fs.ID """
    vessel_weights = fetch_data(query_vessel_weights)
    grouped_vessel_weights = vessel_weights.groupby(['VesselName'])['Weight'].agg(['sum']).reset_index()
    grouped_vessel_weights = grouped_vessel_weights.rename(columns={'sum': 'TotalWeight'})

    vessel_weights['DepartureDate'] = pd.to_datetime(vessel_weights['DepartureDate'])
    vessel_weights = vessel_weights.sort_values(by=['VesselName', 'DepartureDate'])
    vessel_weights['CumulativeWeight'] = vessel_weights.groupby('VesselName')['Weight'].cumsum()


    query_trips_per_boat ="""select Name as VesselName,VesselID,TripNumber from Vessel v join Trip t on v.ID =t.VesselID"""
    df_trips_per_boat = fetch_data(query_trips_per_boat)
    df_trips_per_boat = df_trips_per_boat.groupby("VesselName")["TripNumber"].agg(["count"]).reset_index()
    df_trips_per_boat = df_trips_per_boat.rename(columns={'count': 'NumberOfTrips'})
    

    # Species Group
    query_species ="""select * from FishCatch fc join FishSpecie fs on fc.FishSpecieID = fs.ID  """
    df_species = fetch_data(query_species)
    species_grouped = df_species.groupby('Name')['Weight'].agg(['sum', 'count']).reset_index()
    species_grouped.columns = ['Name', 'TotalWeight', 'Count']
    species_grouped['WeightPercentage'] = (species_grouped['TotalWeight'] / species_grouped['TotalWeight'].sum()) * 100
    threshold = 2  # threshold for grouping small slices
    species_grouped['IsSmall'] = species_grouped['WeightPercentage'] < threshold
    small_slices = species_grouped[species_grouped['IsSmall']]
    large_slices = species_grouped[~species_grouped['IsSmall']]

    if not small_slices.empty:
        other = pd.DataFrame({
            'Name': ['Other'],
            'TotalWeight': [small_slices['TotalWeight'].sum()],
            'WeightPercentage': [small_slices['WeightPercentage'].sum()],
            'IsSmall': [True]
        })
        df_species_combined = pd.concat([large_slices, other], ignore_index=True)
    else:
        df_species_combined = species_grouped

    # Define a function for displaying percentages and values
    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    # Plot 1
    fig1 = px.bar(grouped_vessel_weights, x='VesselName', y='TotalWeight', title='Total Weight Per Vessel')

    # Plot 2
    fig2 = px.line(vessel_weights, x='DepartureDate', y='CumulativeWeight', color='VesselName', markers=True,
                title='Total Weight Per Vessel Over Time')

    # Plot 3
    fig3 = px.pie(
        df_species_combined,
        names='Name',
        values='TotalWeight',
        title='Percentage of Fish Species Caught',
        hole=0.2, 
        labels={'Name': 'Species', 'TotalWeight': 'Weight'},
        color='Name'
    )

    # Plot 4
    fig4 = px.bar(
        df_trips_per_boat,
        x='VesselName',
        y='NumberOfTrips',
        title='Number of Trips per Boat',
        labels={'VesselName': 'Boat', 'NumberOfTrips': 'Number of Trips'},
    )

    # Format the DataFrame to hide repeating values
    formatted_df = format_repeating_values(df)

    today = datetime.now().date()
    
    return html.Div([
        html.H1('Dashboard'),
        html.H3('Daily Report'),
        html.Div([
            dcc.DatePickerRange(
                id='date-picker-range',
                #start_date=df['DepartureDate'].min().date(),  
                start_date=df['DepartureDate'].max().date(), 
                end_date=today,
                #start_date = today,
                #end_date = today,
                display_format='DD/MM/YYYY'
            ),
            html.Div(id='date-filter-output'),
        ]),
        dbc.Row([
            dash_table.DataTable(
                id='formatted-table',
                columns=[{"name": i, "id": i} for i in formatted_df.columns],
                data=formatted_df.to_dict('records'),
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {'if': {'column_id': c}, 'textAlign': 'center'} for c in formatted_df.columns
                ],
                style_table={'overflowX': 'auto',
                            #'height': '500px',
                            'overflowY': 'scroll'},
                #sort_action='native', 
                #filter_action='native'
            ),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3('Number of Vessels',style={"fontSize":"20px"}),
                    html.Div(id='vessels-count', style={
                        'fontSize': '40px',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'lineHeight': '150px',
                        'height': '150px',
                        'width': '150px',
                        'border': '2px solid',
                        'borderRadius': '8px',
                        'backgroundColor': '#f8f9fa',
                        'margin': 'auto'
                    })
                ], className="dbc", style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center'
                }),
            ),
            dbc.Col(
                html.Div([
                    html.H3('Number of Trips',style={"fontSize":"20px"}),
                    html.Div(id='trips-count', style={
                        'fontSize': '40px',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'lineHeight': '150px',
                        'height': '150px',
                        'width': '150px',
                        'border': '2px solid',
                        'borderRadius': '8px',
                        'backgroundColor': '#f8f9fa',
                        'margin': 'auto'
                    })
                ], className="dbc", style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center'
                }),
            ),
            dbc.Col(
                html.Div([
                    html.H3('Total Weigth Caught',style={"fontSize":"20px"}),
                    html.Div(id='total-weigth', style={
                        'fontSize': '40px',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'lineHeight': '150px',
                        'height': '150px',
                        'width': '150px',
                        'border': '2px solid',
                        'borderRadius': '8px',
                        'backgroundColor': '#f8f9fa',
                        'margin': 'auto'
                    })
                ], className="dbc", style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center'
                }),
            ),
            dbc.Col(
                html.Div([
                    html.H3('Total Unloading Weigth',style={"fontSize":"20px"}),
                    html.Div(id='unloading-weight', style={
                        'fontSize': '40px',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'lineHeight': '150px',
                        'height': '150px',
                        'width': '150px',
                        'border': '2px solid',
                        'borderRadius': '8px',
                        'backgroundColor': '#f8f9fa',
                        'margin': 'auto'
                    })
                ], className="dbc", style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center'
                }),
            ),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig1), width=6),  # Adjust width as needed
            dbc.Col(dcc.Graph(figure=fig2), width=6),  # Adjust width as needed
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig4), width=6),  # Adjust width as needed
            dbc.Col(dcc.Graph(figure=fig3), width=6),  # Adjust width as needed
        ]),
    ],className='dbc'),

#Callback to display the number of vessels
@app.callback(
    Output('vessels-count', 'children'),
    [Input('vessels-count', 'id')]
)
def update_vessel_count(_):
    query = """
    select count(ID) as number_of_vessels from Vessel where Active = 1 and DeletedDate is NULL
    """
    df = fetch_data(query)
    number_of_vessels = df['number_of_vessels'][0]
    return f'{number_of_vessels}'

#Callback to display the total weight caught
@app.callback(
    Output('total-weigth', 'children'),
    [Input('total-weigth', 'id')]
)
def update_total_weigth(_):
    query="""
    select sum(Weight) as TotalWeight from FishCatch group by ID
    """
    df_total_weights = fetch_data(query)
    total_weight = int(df_total_weights["TotalWeight"].sum())
    return f'{total_weight} t'

#Callback to display the number of trips
@app.callback(
    Output('trips-count', 'children'),
    [Input('trips-count', 'id')]
)
def update_number_of_trips(_):
    query="""
    select * from Vessel v join Trip t on v.ID = t.VesselID
    """
    df_trips = fetch_data(query)
    df_trips = df_trips.groupby('Name')['TripNumber'].nunique().reset_index()
    number_of_trips = df_trips["TripNumber"].sum()
    return f'{number_of_trips}'

#Callback to display the total unloading weight
@app.callback(
    Output('unloading-weight', 'children'),
    [Input('unloading-weight', 'id')]
)
def update_unloading_weight(_):
    query="""
    select sum(Weight) as TotalUnloadingWeight from UnloadingData group by ID
    """
    df_unloadings_data = fetch_data(query)
    unloading_weight = int(df_unloadings_data["TotalUnloadingWeight"].sum())
    return f'{unloading_weight} t'
    
    
    
@app.callback(
    Output('formatted-table', 'data'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_table(start_date, end_date):

    if not start_date or not end_date:
        return [] 

    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d %H:%M:%S')

    query = text("""
    SELECT DISTINCT
        VesselName, 
        TripNumber, 
        DepartureDate, 
        ArrivalDate, 
        FishingSetNumber, 
        FishSpecieName, 
        FishSize, 
        SUM(SUM(TotalWeight)) OVER (PARTITION BY FishSpecieName, FishSize, FishingSetID) AS TotalWeightPerSize, 
        SUM(SUM(TotalWeight)) OVER (PARTITION BY FishSpecieName, FishingSetID) AS TotalWeightPerSpecies, 
        SUM(SUM(TotalWeight)) OVER (PARTITION BY FishingSetID) AS TotalWeightPerSet, 
        SUM(SUM(TotalWeight)) OVER (PARTITION BY TripNumber) AS TotalWeightPerTrip,
                 Port,
        Pier, 
        SUM(SUM(UnloadingWeight)) OVER (PARTITION BY TripNumber) AS TotalUnloadingWeight,
        ROUND(CASE WHEN SUM(SUM(UnloadingWeight)) OVER (PARTITION BY TripNumber) = 0 THEN 0 ELSE SUM(SUM(TotalWeight)) OVER (PARTITION BY TripNumber) / SUM(SUM(UnloadingWeight)) OVER (PARTITION BY TripNumber) END, 2) AS 'P/D'
    FROM ( 
        SELECT 
            v.Name AS VesselName, 
            t.TripNumber, 
            t.DateStart AS DepartureDate, 
            t.DateEnd AS ArrivalDate, 
            fs.Number AS FishingSetNumber, 
            s.Name AS FishSpecieName, 
            fsize.Name AS FishSize, 
            fc.Weight AS TotalWeight, 
            fs.ID AS FishingSetID,
            p.Name AS Port, 
            pier.Name AS Pier, 
            ud.Weight AS UnloadingWeight 
        FROM 
            Vessel v 
            JOIN Trip t ON v.ID = t.VesselID 
            JOIN FishingZone fz ON t.TripNumber = fz.TripNumber AND v.ID = fz.VesselID 
            JOIN FishingSet fs ON fs.FishingZoneID = fz.ID 
            JOIN FishCatch fc ON fc.FishingSetID = fs.ID 
            JOIN FishSpecie s ON s.ID = fc.FishSpecieID 
            JOIN FishSize fsize ON fsize.ID = fc.FishSizeID
            LEFT JOIN Unloading u ON t.TripNumber = u.TripNumber AND v.ID = u.VesselID 
            LEFT JOIN UnloadingData ud ON u.ID = ud.UnloadingID 
            LEFT JOIN Port p on u.PortID = p.ID 
            LEFT JOIN Pier pier on u.PierID = pier.ID 
    ) AS subquery 
    WHERE ((DepartureDate <= :end_date AND ArrivalDate >= :start_date) OR (DepartureDate >= :start_date))
    GROUP BY 
        VesselName, 
        TripNumber, 
        DepartureDate, 
        ArrivalDate, 
        FishingSetNumber, 
        FishSpecieName, 
        FishSize, 
        FishingSetID,
        Port,
        Pier,
        UnloadingWeight
    ORDER BY 
        VesselName, 
        TripNumber, 
        FishingSetNumber, 
        FishSpecieName, 
        FishSize
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={'start_date': start_date, 'end_date': end_date})
        df = format_repeating_values(df)

    # Ensure date columns are recognized as datetime types
    df['DepartureDate'] = pd.to_datetime(df['DepartureDate'])
    df['ArrivalDate'] = pd.to_datetime(df['ArrivalDate'])

    formatted_df = format_repeating_values(df)
    
    return formatted_df.to_dict('records')

server = app.server

if __name__ == '__main__':
    app.run_server(debug=False)
