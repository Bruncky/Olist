'''
Order module to filter out orders from dataset
'''

import datetime
import pandas as pd

from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders delivered as index,
    and various properties of these orders as columns
    '''

    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()


    def get_wait_time(self, is_delivered = True):
        """
        02-01 > Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected]
        and filtering out non-delivered orders unless specified
        """

        # Grabbing orders by using initializer logic
        orders = self.data['orders'].copy()

        # Filter delivered orders
        if is_delivered:
            orders = orders.query("order_status == 'delivered'").copy()

        # Converting every relevant column to datetime
        orders['order_purchase_timestamp'] =\
            pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date'] =\
            pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] =\
            pd.to_datetime(orders['order_estimated_delivery_date'])

        one_day_delta = datetime.timedelta(days = 1)

        # Computing wait time and adding it to the DF
        wait_time = orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']
        orders.loc[:, 'wait_time'] = wait_time / one_day_delta

        # Computing delay vs. expected and adding it to the DF
        delay_vs_expected =\
            orders['order_estimated_delivery_date'] - orders['order_delivered_customer_date']
        orders.loc[:, 'delay_vs_expected'] = delay_vs_expected / one_day_delta

        # Now we handle the delays - some of them are negative and we don't want that
        def absolute_delay(delay):
            if delay < 0:
                return abs(delay)

            return 0

        # This line locates the 'delay_vs_expected' column and applies the method to every line
        orders.loc[:, 'delay_vs_expected'] = orders['delay_vs_expected'].apply(absolute_delay)

        # Computing expected wait time and adding it to the DF
        expected_wait_time =\
            orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']
        orders.loc[:, 'expected_wait_time'] = expected_wait_time / one_day_delta

        return orders[['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected']]


    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """

        # Grabbing reviews by using initializer logic
        reviews = self.data['order_reviews'].copy()

        # Checking whether the score is five or one
        def dim_five_star(score):
            if score == 5:
                return 1

            return 0

        def dim_one_star(score):
            if score == 1:
                return 1

            return 0

        reviews['dim_is_five_star'] = reviews['review_score'].map(dim_five_star)
        reviews['dim_is_one_star'] = reviews['review_score'].map(dim_one_star)

        return reviews[['order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score']]


    def get_number_products(self):
        """
        Returns a DataFrame with:
        order_id, number_of_products
        """

        # Summarizing 'order_items' by order_id, thus counting number of products per order
        order_items = self.data["order_items"].copy()

        return order_items.groupby("order_id").count()\
                    .sort_values('order_item_id')\
                    .rename(columns = {'order_item_id': 'number_of_products'})\
                    [['number_of_products']]


    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """

        # Summarizing order_items by order_id, then we can count sellers per order
        # Line below inspired by solution. After grouping, we count unique sellers per order
        # and reset the index to get a normal DF
        sellers = self.data['order_items'].groupby('order_id')['seller_id'].nunique().reset_index()

        return sellers.sort_values('seller_id').rename(columns = {'seller_id': 'number_of_sellers'})


    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """

        return self.data['order_items'].groupby('order_id')[['price', 'freight_value']].sum()


    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with order_id
        and distance_seller_customer
        """

        # import data
        data = self.data
        matching_table = Olist().get_matching_table()

        # Since one zip code can map to multiple (lat, lng), take the first one
        geo = data['geolocation']
        geo = geo.groupby('geolocation_zip_code_prefix', as_index = False).first()

        # Select sellers and customers
        sellers = data['sellers']
        customers = data['customers']

        # Merge geo_location for sellers
        sellers_mask_columns = ['seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state', 'geolocation_lat', 'geolocation_lng']

        sellers_geo = sellers.merge(geo, how = 'left', left_on = 'seller_zip_code_prefix', right_on = 'geolocation_zip_code_prefix')[sellers_mask_columns]

        # Merge geo_location for customers
        customers_mask_columns = ['customer_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state', 'geolocation_lat', 'geolocation_lng']

        customers_geo = customers.merge(geo, how = 'left', left_on = 'customer_zip_code_prefix', right_on = 'geolocation_zip_code_prefix')[customers_mask_columns]

        # Use the matching table and merge customers and sellers
        matching_geo = matching_table.merge(sellers_geo, on = 'seller_id')\
            .merge(customers_geo, on = 'customer_id', suffixes = ('_seller', '_customer'))

        # Remove na()
        matching_geo = matching_geo.dropna()

        matching_geo.loc[:, 'distance_seller_customer'] = matching_geo.apply(
            lambda row: haversine_distance(
                row['geolocation_lng_seller'],
                row['geolocation_lat_seller'],
                row['geolocation_lng_customer'],
                row['geolocation_lat_customer']
            ),
            axis = 1
        )

        # Since an order can have multiple sellers,
        # return the average of the distance per order
        order_distance = matching_geo.groupby(
            'order_id',
            as_index = False
        ).agg({'distance_seller_customer': 'mean'})

        return order_distance

    def get_training_data(self, is_delivered = True, with_distance_seller_customer = False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_products', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """

        training_set = self.get_wait_time(is_delivered)\
            .merge(self.get_review_score(), on = 'order_id')\
            .merge(self.get_number_products(), on = 'order_id')\
            .merge(self.get_number_sellers(), on = 'order_id')\
            .merge(self.get_price_and_freight(), on = 'order_id')

        # Skip heavy computation of distance_seller_customer unless specified
        if with_distance_seller_customer:
            training_set = training_set.merge(self.get_distance_seller_customer(), on = 'order_id')

        # .dropna() removes NaN values
        return training_set.dropna()
