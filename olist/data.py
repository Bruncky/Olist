"""
Data module for reading data from CSVs
"""

import os
import pandas as pd

class Olist:
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """

        root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        csv_path = os.path.join(root_path, 'data', 'csv')
        
        data = {}
        file_names = os.listdir(csv_path)

        def key_from_file_name(filename):
            if filename == '.DS_Store':
                os.remove(filename)

            return filename.replace('olist_', '').replace('_dataset.csv', '').replace('.csv', '')

        for file in file_names:
            data[key_from_file_name(file)] = pd.read_csv(f'{csv_path}/{file}')

        return data


    def get_matching_table(self):
        """
        This function returns a matching table between
        columns [ "order_id", "review_id", "customer_id", "product_id", "seller_id"]
        """

        data = self.get_data()

        orders = data['orders'][['customer_id', 'order_id']]
        items = data['order_items'][['order_id', 'product_id', 'seller_id']]
        reviews = data['order_reviews'][['order_id', 'review_id']]

        matching_table = orders\
            .merge(reviews, on = 'order_id', how = 'outer')\
            .merge(items, on = 'order_id', how = 'outer')

        return matching_table
