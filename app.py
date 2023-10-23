from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
ignore_variables = [
    'Order',
    'PID',
]

def plot_numericals(data, cols):
    summary = data[cols] \
        .describe() \
        .transpose() \
        .sort_values(by='count')


    n = data.shape[0]
    b = int(np.sqrt(n))
    for k, (col, val) in enumerate(summary['count'].items()):
        plt.figure()
        data[col].plot.hist(bins=b)
        plt.title(f'{col}, n={int(val)}')
        plt.show()

# plot_numericals(data, data.select_dtypes('number').columns)

from typing import Tuple

def remap_categories(
    series: pd.Series,
    old_categories: Tuple[str],
    new_category: str,
) -> pd.Series:
    # Add the new category to the list of valid categories.
    series = series.cat.add_categories(new_category)

    # Set all items of the old categories as the new category.
    remapped_items = series.isin(old_categories)
    series.loc[remapped_items] = new_category

    # Clean up the list of categories, the old categories no longer exist.
    series = series.cat.remove_unused_categories()

    return series

@app.route('/predict')
def predict():
    ignore_variables = [
        'Order',
        'PID',
    ]

    continuous_variables = [
        'Lot.Frontage',
        'Lot.Area',
        'Mas.Vnr.Area',
        'BsmtFin.SF.1',
        'BsmtFin.SF.2',
        'Bsmt.Unf.SF',
        'Total.Bsmt.SF',
        'X1st.Flr.SF',
        'X2nd.Flr.SF',
        'Low.Qual.Fin.SF',
        'Gr.Liv.Area',
        'Garage.Area',
        'Wood.Deck.SF',
        'Open.Porch.SF',
        'Enclosed.Porch',
        'X3Ssn.Porch',
        'Screen.Porch',
        'Pool.Area',
        'Misc.Val',
        
    ]

    discrete_variables = [
        'Year.Built',
        'Year.Remod.Add',
        'Bsmt.Full.Bath',
        'Bsmt.Half.Bath',
        'Full.Bath',
        'Half.Bath',
        'Bedroom.AbvGr',
        'Kitchen.AbvGr',
        'TotRms.AbvGrd',
        'Fireplaces',
        'Garage.Yr.Blt',
        'Garage.Cars',
        'Mo.Sold',
        'Yr.Sold',
    ]

    ordinal_variables = [
        'Lot.Shape',
        'Utilities',
        'Land.Slope',
        'Overall.Qual',
        'Overall.Cond',
        'Exter.Qual',
        'Exter.Cond',
        'Bsmt.Qual',
        'Bsmt.Cond',
        'Bsmt.Exposure',
        'BsmtFin.Type.1',
        'BsmtFin.Type.2',
        'Heating.QC',
        'Electrical',
        'Kitchen.Qual',
        'Functional',
        'Fireplace.Qu',
        'Garage.Finish',
        'Garage.Qual',
        'Garage.Cond',
        'Paved.Drive',
        'Pool.QC',
        'Fence',
    ]

    categorical_variables = [
        'MS.SubClass',
        'MS.Zoning',
        'Street',
        'Alley',
        'Land.Contour',
        'Lot.Config',
        'Neighborhood',
        'Condition.1',
        'Condition.2',
        'Bldg.Type',
        'House.Style',
        'Roof.Style',
        'Roof.Matl',
        'Exterior.1st',
        'Exterior.2nd',
        'Mas.Vnr.Type',
        'Foundation',
        'Heating',
        'Central.Air',
        'Garage.Type',
        'Misc.Feature',
        'Sale.Type',
        'Sale.Condition',
    ]

    data = request.get_json()
    data = [data]
    molde = 0
    with open('molde.json', 'r') as f:
        f = f.read()
        molde = json.loads(f)
  

    molde = pd.DataFrame([molde])
   
    # Criando um DataFrame com os dados do arquivo data.json
    data = pd.DataFrame(data)
    data.drop(columns=['Order', 'PID'], inplace=True)

    

    
    for col in continuous_variables:
        data[col] = data[col].astype('float64')
    for col in categorical_variables:
        data[col] = data[col].astype('category')

    for col in discrete_variables:
        data[col] = data[col].astype('float64')


    category_orderings = {
        'Lot.Shape': [
            'Reg',
            'IR1',
            'IR2',
            'IR3',
        ],
        'Utilities': [
            'AllPub',
            'NoSewr',
            'NoSeWa',
            'ELO',
        ],
        'Land.Slope': [
            'Gtl',
            'Mod',
            'Sev',
        ],
        'Overall.Qual': [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ],
        'Overall.Cond': [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ],
        'Exter.Qual': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Exter.Cond': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Bsmt.Qual': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Bsmt.Cond': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Bsmt.Exposure': [
            'Gd',
            'Av',
            'Mn',
            'No',
            'NA',
        ],
        'BsmtFin.Type.1': [
            'GLQ',
            'ALQ',
            'BLQ',
            'Rec',
            'LwQ',
            'Unf',
        ],
        'BsmtFin.Type.2': [
            'GLQ',
            'ALQ',
            'BLQ',
            'Rec',
            'LwQ',
            'Unf',
        ],
        'Heating.QC': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Electrical': [
            'SBrkr',
            'FuseA',
            'FuseF',
            'FuseP',
            'Mix',
        ],
        'Kitchen.Qual': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Functional': [
            'Typ',
            'Min1',
            'Min2',
            'Mod',
            'Maj1',
            'Maj2',
            'Sev',
            'Sal',
        ],
        'Fireplace.Qu': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Garage.Finish': [
            'Fin',
            'RFn',
            'Unf',
        ],
        'Garage.Qual': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',
        ],
        'Garage.Cond': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
            'Po',    
        ],
        'Paved.Drive': [
            'Y',
            'P',
            'N',
        ],
        'Pool.QC': [
            'Ex',
            'Gd',
            'TA',
            'Fa',
        ],
        'Fence': [
            'GdPrv',
            'MnPrv',
            'GdWo',
            'MnWw',
        ],
    }


    for col, orderings in category_orderings.items():
        data[col] = data[col] \
            .astype('category') \
            .cat \
            .set_categories(orderings, ordered=True)
        
    
    with open("processado.pkl", 'wb') as file:
        pickle.dump(
            [
                data,
                continuous_variables,
                discrete_variables,
                ordinal_variables,
                categorical_variables,
            ],
            file,
    )
        

    with open("processado.pkl", 'rb') as file:
        (
            data,
            continuous_variables,
            discrete_variables,
            ordinal_variables,
            categorical_variables,
        ) = pickle.load(file)
  
    data['MS.Zoning'].unique()

    selection = ~(data['MS.Zoning'].isin(['A (agr)', 'C (all)', 'I (all)']))

    data = data[selection]
    data['MS.Zoning'] = data['MS.Zoning'].cat.remove_unused_categories()
    data['Sale.Type'].unique()

    processed_data = data.copy()

    processed_data['Sale.Type'] = remap_categories(
    series=processed_data['Sale.Type'],
    old_categories=('WD ', 'CWD', 'VWD'),
    new_category='GroupedWD',
    )

    processed_data['Sale.Type'] = remap_categories(
        series=processed_data['Sale.Type'],
        old_categories=('COD', 'ConLI', 'Con', 'ConLD', 'Oth', 'ConLw'),
        new_category='Other',
    )

    data = processed_data

    data = data.drop(columns='Street')
    pd.crosstab(data['Condition.1'], data['Condition.2'])
    processed_data = data.copy()

    for col in ('Condition.1', 'Condition.2'):
        processed_data[col] = remap_categories(
            series=processed_data[col],
            old_categories=('RRAn', 'RRAe', 'RRNn', 'RRNe'),
            new_category='Railroad',
        )
        processed_data[col] = remap_categories(
            series=processed_data[col],
            old_categories=('Feedr', 'Artery'),
            new_category='Roads',
        )
        processed_data[col] = remap_categories(
            series=processed_data[col],
            old_categories=('PosA', 'PosN'),
            new_category='Positive',
        )

    pd.crosstab(processed_data['Condition.1'], processed_data['Condition.2'])

    processed_data['Condition'] = pd.Series(
    index=processed_data.index,
    dtype=pd.CategoricalDtype(categories=(
        'Norm',
        'Railroad',
        'Roads',
        'Positive',
        'RoadsAndRailroad',
        )),
    )

    norm_items = processed_data['Condition.1'] == 'Norm'
    processed_data['Condition'][norm_items] = 'Norm'

    railroad_items = \
    (processed_data['Condition.1'] == 'Railroad') \
    & (processed_data['Condition.2'] == 'Norm')
    processed_data['Condition'][railroad_items] = 'Railroad'

    roads_items = \
    (processed_data['Condition.1'] == 'Roads') \
    & (processed_data['Condition.2'] != 'Railroad')
    processed_data['Condition'][roads_items] = 'Roads'
    positive_items = processed_data['Condition.1'] == 'Positive'
    processed_data['Condition'][positive_items] = 'Positive'
    roads_and_railroad_items = \
    ( \
        (processed_data['Condition.1'] == 'Railroad') \
        & (processed_data['Condition.2'] == 'Roads')
    ) \
    | ( \
        (processed_data['Condition.1'] == 'Roads') \
        & (processed_data['Condition.2'] == 'Railroad') \
    )
    processed_data['Condition'][roads_and_railroad_items] = 'RoadsAndRailroad'

    processed_data = processed_data.drop(columns=['Condition.1', 'Condition.2'])

    data = processed_data

    data['HasShed'] = data['Misc.Feature'] == 'Shed'
    data = data.drop(columns='Misc.Feature')

    data['HasAlley'] = ~data['Alley'].isna()
    data = data.drop(columns='Alley')

    data['Exterior.2nd'] = remap_categories(
    series=data['Exterior.2nd'],
    old_categories=('Brk Cmn', ),
    new_category='BrkComm',
    )
    data['Exterior.2nd'] = remap_categories(
        series=data['Exterior.2nd'],
        old_categories=('CmentBd', ),
        new_category='CemntBd',
    )
    data['Exterior.2nd'] = remap_categories(
        series=data['Exterior.2nd'],
        old_categories=('Wd Shng', ),
        new_category='WdShing',
    )

    for col in ('Exterior.1st', 'Exterior.2nd'):
        categories = data[col].cat.categories
        data[col] = data[col].cat.reorder_categories(sorted(categories))

    processed_data = data.copy()
    mat_count = processed_data['Exterior.1st'].value_counts()
    rare_materials = list(mat_count[mat_count < 40].index)
    processed_data['Exterior'] = remap_categories(
    series=processed_data['Exterior.1st'],
    old_categories=rare_materials,
    new_category='Other',
)
    processed_data = processed_data.drop(columns=['Exterior.1st', 'Exterior.2nd'])
    data = processed_data
    data = data.drop(columns='Heating')
    data = data.drop(columns='Roof.Matl')

    data['Roof.Style'] = remap_categories(
    series=data['Roof.Style'],
    old_categories=[
        'Flat',
        'Gambrel',
        'Mansard',
        'Shed',
    ],
    new_category='Other',
    )

    data['Mas.Vnr.Type'] = remap_categories(
    series=data['Mas.Vnr.Type'],
    old_categories=[
        'BrkCmn',
        'CBlock',
    ],
    new_category='Other',
    )

    data['Mas.Vnr.Type'] = data['Mas.Vnr.Type'].cat.add_categories('None')
    data['Mas.Vnr.Type'][data['Mas.Vnr.Type'].isna()] = 'None'

    data['MS.SubClass'] = remap_categories(
    series=data['MS.SubClass'],
    old_categories=[75, 45, 180, 40, 150],
    new_category='Other',
    )
    
    data['Foundation'] = remap_categories(
    series=data['Foundation'],
    old_categories=['Slab', 'Stone', 'Wood'],
    new_category='Other',
    )
    

    selection = ~data['Neighborhood'].isin([
    'Blueste',
    'Greens',
    'GrnHill',
    'Landmrk',
    ])
    data = data[selection]

    data['Neighborhood'] = data['Neighborhood'].cat.remove_unused_categories()

    data['Garage.Type'] = data['Garage.Type'].cat.add_categories(['NoGarage'])
    data['Garage.Type'][data['Garage.Type'].isna()] = 'NoGarage'
    all_categorical = data.select_dtypes('category').columns

    new_categorical_variables = [ \
        col for col in all_categorical \
        if not col in ordinal_variables \
    ]

    data = data.drop(columns='Utilities')

    data = data.drop(columns='Pool.QC')

    data['Fence'].value_counts().sort_index()

    old_categories = list(data['Fence'].cat.categories)

    new_categories = old_categories + ['NoFence']

    data['Fence'] = data['Fence'].cat.set_categories(new_categories)

    data['Fence'][data['Fence'].isna()] = 'NoFence'

    data = data.drop(columns='Fireplace.Qu')
    data = data.drop(columns=['Garage.Cond', 'Garage.Qual'])
    data['Garage.Finish'] = data['Garage.Finish'] \
    .cat \
    .as_unordered() \
    .cat \
    .add_categories(['NoGarage'])
    data['Garage.Finish'][data['Garage.Finish'].isna()] = 'NoGarage'

    data['Electrical'][data['Electrical'].isna()] = 'SBrkr'
    ordinal_columns = [col for col in data.select_dtypes('category') if data[col].cat.ordered]

    data['Bsmt.Exposure'][data['Bsmt.Exposure'].isna()] = 'NA'
    data['Bsmt.Exposure'] = data['Bsmt.Exposure'] \
        .cat \
        .as_unordered() \
        .cat \
        .remove_unused_categories()
    
    for col in ('Bsmt.Qual', 'Bsmt.Cond', 'BsmtFin.Type.1', 'BsmtFin.Type.2'):
        data[col] = data[col].cat.add_categories(['NA'])
        data[col][data[col].isna()] = 'NA'
        data[col] = data[col] \
            .cat \
            .as_unordered() \
            .cat \
            .remove_unused_categories()
    
    data['Bsmt.Cond'][data['Bsmt.Cond'] == 'Po'] = 'Fa'
    data['Bsmt.Cond'][data['Bsmt.Cond'] == 'Ex'] = 'Gd'
    data['Bsmt.Cond'] = data['Bsmt.Cond'].cat.remove_unused_categories()
   
    missing_lot_frontage = data['Lot.Frontage'].isna()
    aux_data = data[['Lot.Frontage', 'Lot.Area']].copy()
    aux_data['Sqrt.Lot.Area'] = aux_data['Lot.Area'].apply(np.sqrt)
    data['Lot.Frontage'] = data['Lot.Frontage'].fillna(data['Lot.Frontage'].median())
    garage_age = data['Yr.Sold'] - data['Garage.Yr.Blt']
    data[garage_age < 0.0].transpose()
    garage_age[garage_age < 0.0] = 0.0
    data = data.drop(columns='Garage.Yr.Blt')
    data['Garage.Age'] = garage_age
    data['Garage.Age'] = data['Garage.Age'].fillna(data['Garage.Age'].median())
    remod_age = data['Yr.Sold'] - data['Year.Remod.Add']
    data[remod_age < 0.0].transpose()
    remod_age[remod_age < 0.0] = 0.0
    house_age = data['Yr.Sold'] - data['Year.Built']
    data[house_age < 0.0].transpose()
    house_age[house_age < 0.0] = 0.0
    data = data.drop(columns=['Year.Remod.Add', 'Year.Built'])
    data['Remod.Age'] = remod_age
    data['House.Age'] = house_age

    data.loc[data['Mas.Vnr.Area'].isna(), 'Mas.Vnr.Area'] = 0.0
   
    data = data.dropna(axis=0)
    for col in data.select_dtypes('category').columns:
        data[col] = data[col].cat.remove_unused_categories()

  
 
    categorical_columns = data.select_dtypes('category').columns
    model_data = data.copy()

    
 

    categorical_columns = []
    ordinal_columns = []
    for col in model_data.select_dtypes('category').columns:
        if model_data[col].cat.ordered:
            ordinal_columns.append(col)
        else:
            categorical_columns.append(col)
    
    for col in ordinal_columns:
        codes, _ = pd.factorize(data[col], sort=True)
        model_data[col] = codes

    
    original_data = model_data['Exterior']
    encoded_data = pd.get_dummies(original_data)

    aux_dataframe = encoded_data
    aux_dataframe['Exterior'] = original_data.copy()
    original_data = model_data['Exterior']
    encoded_data = pd.get_dummies(original_data, drop_first=True)

    aux_dataframe = encoded_data
    aux_dataframe['Exterior'] = original_data.copy()

    model_data = pd.get_dummies(model_data, drop_first=True)

    model_data['TotalSF'] = model_data['X1st.Flr.SF'] + model_data['X2nd.Flr.SF'] + model_data['Total.Bsmt.SF']

    model_data['TotalBath'] = model_data['Full.Bath'] + model_data['Half.Bath'] + model_data['Bsmt.Full.Bath'] +  model_data['Bsmt.Half.Bath']
    model_data.drop(['Total.Bsmt.SF', 'X2nd.Flr.SF','X1st.Flr.SF'], axis=1, inplace=True)

    model_data = model_data.loc[model_data["Gr.Liv.Area"] < 3500]

   

    aplicar_log = ["Wood.Deck.SF", "Open.Porch.SF", "Garage.Age", "Remod.Age", "Mas.Vnr.Area", "Bsmt.Unf.SF", "BsmtFin.SF.1", "House.Age"] ## 
    model_data[aplicar_log] = model_data[aplicar_log].apply(np.log1p)
    remover  = ['Misc.Val', 'Pool.Area', 'Screen.Porch', 'X3Ssn.Porch', 'X3Ssn.Porch', 'Enclosed.Porch','BsmtFin.SF.2','Low.Qual.Fin.SF']
    model_data.drop(columns=remover, inplace=True)
    
    na_duvida = [ "Functional", "Paved.Drive", "Land.Slope", "Lot.Shape",  "Exter.Cond", "Electrical"] ## 'X2nd.Flr.SF', "Bsmt.Half.Bath"
    model_data.drop(columns=na_duvida, inplace=True)
 
    
    for colum in model_data.columns:
     
       molde[colum] = model_data[colum]
   
    
    model = joblib.load('preditor_de_casa.pkl')
    res = model.predict(molde)
    return jsonify({'preco': 10**res[0]})

if __name__ == '__main__':
    app.run(debug=True)