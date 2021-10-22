import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Product:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.matching_table = olist.get_matching_table()
        self.order = Order()

    def get_product_features(self):
        """
        Returns a DataFrame with:
        'product_id', 'product_category_name', 'product_name_length',
        'product_description_length', 'product_photos_qty', 'product_weight_g',
        'product_length_cm', 'product_height_cm', 'product_width_cm'
        """
        pass  # YOUR CODE HERE

    def get_price(self):
        """
        Return a DataFrame with:
        'product_id', 'price'
        """
        pass  # YOUR CODE HERE

    def get_wait_time(self):
        """
        Returns a DataFrame with:
        'product_id', 'wait_time'
        """
        pass  # YOUR CODE HERE

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'product_id', 'share_of_five_stars', 'share_of_one_stars',
        'review_score'
        """
        pass  # YOUR CODE HERE

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'product_id', 'n_orders', 'quantity'
        """
        pass  # YOUR CODE HERE

    def get_sales(self):
        """
        Returns a DataFrame with:
        'product_id', 'sales'
        """
        pass  # YOUR CODE HERE

    def get_training_data(self):
        """
        Returns a DataFrame with:
        ['product_id', 'product_name_length', 'product_description_length',
        'product_photos_qty', 'product_weight_g', 'product_length_cm',
        'product_height_cm', 'product_width_cm', 'category', 'wait_time',
        'price', 'share_of_one_stars', 'share_of_five_stars', 'review_score',
        'n_orders', 'quantity', 'sales'],
        """
        pass  # YOUR CODE HERE

    def get_product_cat(self, agg="median"):
        '''
        Returns a DataFrame with `category` as index, and aggregating various properties for each category in columns such as:
        - `quantity`: total number of products sold for this category.
        - `product_weight_g`: mean or median weight per category
        - ...
        '''
        pass  # YOUR CODE HERE
