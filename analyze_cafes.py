import pandas as pd
import numpy as np
import json
import os

# Set relative path to data
DATA_PATH = os.path.join('data', 'zomato.csv')

def process_data():
    print("Loading data...")
    # Load only necessary columns for the dashboard
    cols = ['name', 'online_order', 'book_table', 'rate', 'votes', 'location',
            'rest_type', 'dish_liked', 'cuisines', 'approx_cost(for two people)', 'address']

    # Read the data
    df = pd.read_csv(DATA_PATH, usecols=cols)

    # Data Cleaning
    print("Cleaning data...")
    df = df.dropna(subset=['rate', 'approx_cost(for two people)', 'location'])

    # Clean rate - remove NEW, -, and anything that can't be converted
    df['rate'] = df['rate'].astype(str).str.strip()
    df = df[~df['rate'].isin(['NEW', '-', 'nan', ''])]
    df['rate'] = df['rate'].str.split('/').str[0].str.strip()
    df = df[df['rate'].str.replace('.', '', 1).str.isnumeric()]
    df['rate'] = df['rate'].astype(float)

    # Clean cost
    df['approx_cost(for two people)'] = (
        df['approx_cost(for two people)']
        .astype(str)
        .str.replace(',', '')
        .str.strip()
        .astype(float)
    )

    # Clean rest_type and cuisines
    df['rest_type'] = df['rest_type'].fillna('Unknown')
    df['cuisines'] = df['cuisines'].fillna('Unknown')
    df['dish_liked'] = df['dish_liked'].fillna('N/A')

    # Clean column name
    df = df.rename(columns={'approx_cost(for two people)': 'cost_for_two'})

    print(f"Data cleaned. {len(df)} records loaded.")
    return df


def get_top_locations(df, n=15):
    """Top locations by average rating and number of restaurants"""
    loc_stats = df.groupby('location').agg(
        avg_rating=('rate', 'mean'),
        num_restaurants=('name', 'count'),
        avg_cost=('cost_for_two', 'mean')
    ).reset_index()
    loc_stats = loc_stats[loc_stats['num_restaurants'] >= 10]
    return loc_stats.sort_values('avg_rating', ascending=False).head(n)


def get_cuisine_popularity(df, n=15):
    """Most popular cuisines by count"""
    cuisines = df['cuisines'].str.split(',').explode().str.strip()
    cuisine_counts = cuisines.value_counts().head(n).reset_index()
    cuisine_counts.columns = ['cuisine', 'count']
    return cuisine_counts


def get_cost_vs_rating(df):
    """Cost vs rating for scatter analysis"""
    sample = df[['name', 'rate', 'cost_for_two', 'location', 'cuisines', 'votes']].dropna()
    return sample.sample(min(1000, len(sample)), random_state=42)


def get_online_order_stats(df):
    """Online order and table booking trends"""
    stats = {
        'online_order': df['online_order'].value_counts().to_dict(),
        'book_table': df['book_table'].value_counts().to_dict(),
        'online_vs_rating': df.groupby('online_order')['rate'].mean().to_dict(),
        'booking_vs_rating': df.groupby('book_table')['rate'].mean().to_dict(),
    }
    return stats


def get_value_for_money(df, n=20):
    """Best value cafes: high rating, low cost, high votes"""
    df = df.copy()
    df['value_score'] = (df['rate'] / df['cost_for_two']) * np.log1p(df['votes'])
    top = df.nlargest(n, 'value_score')[['name', 'location', 'rate', 'cost_for_two', 'votes', 'cuisines', 'value_score']]
    return top


def get_rating_distribution(df):
    """Rating distribution for histogram"""
    return df['rate'].dropna()


def get_rest_type_stats(df, n=10):
    """Restaurant type breakdown"""
    rt = df['rest_type'].str.split(',').str[0].str.strip()
    return rt.value_counts().head(n).reset_index().rename(columns={'index': 'rest_type', 'rest_type': 'count'})


if __name__ == '__main__':
    df = process_data()
    print("\n--- Quick Stats ---")
    print(f"Total restaurants: {len(df)}")
    print(f"Unique locations: {df['location'].nunique()}")
    print(f"Average rating: {df['rate'].mean():.2f}")
    print(f"Average cost for two: ₹{df['cost_for_two'].mean():.0f}")
    print(f"\nTop 5 locations by avg rating:")
    print(get_top_locations(df).head(5)[['location', 'avg_rating', 'num_restaurants']].to_string(index=False))
    print(f"\nTop 5 cuisines:")
    print(get_cuisine_popularity(df).head(5).to_string(index=False))
    print(f"\nTop 5 value-for-money cafes:")
    print(get_value_for_money(df).head(5)[['name', 'location', 'rate', 'cost_for_two']].to_string(index=False))
