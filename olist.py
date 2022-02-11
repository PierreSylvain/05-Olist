import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import ward, fcluster

from sklearn import metrics
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from dateutil.relativedelta import relativedelta
sns.set()
palette = sns.color_palette("bright", 4)

class Olist:
    """Manage data from Olist
    """

    def __init__(self):
        self.nb_cluster = 3
        self.scaler = scaler = MinMaxScaler()
        self.default_cluster = 1
        self.base_directory = '../data/'
        self.features = [
            'order_purchase_delivered_minutes',
            'payment_value',
            'Recency',
            'review_score',
            'Frequency',
            'Monetary'
        ]

    def R_Class(self, r_value, col, quantiles):
        """Recency. set Recency classification by quantille.
        from old (1) to recent (4)
        Arguments:
            r_value {float} -- Recency value
            col {string} -- column name for Recency
            quantiles {dict} -- Quantiles dict

        Returns:
            int -- Recency classification
        """
        if r_value <= quantiles[col][0.25]:
            return 4
        elif r_value <= quantiles[col][0.50]:
            return 3
        elif r_value <= quantiles[col][0.75]:
            return 2
        else:
            return 1

    def FM_Class(self, fm_value, col, quantiles):
        """Frequency or Monetary. set Frequency or Monetary classification by quantille.
        from high (1) to low (4)
        Arguments:
            fm_value {float} -- Frequency or Monetary value
            col {string} -- Column name
            quantiles {dict} -- Quantiles dict

        Returns:
            int -- Recency classification
        """
        if fm_value <= quantiles[col][0.25]:
            return 1
        elif fm_value <= quantiles[col][0.50]:
            return 2
        elif fm_value <= quantiles[col][0.75]:
            return 3
        else:
            return 4

    def get_RFM(self, df, reference_date):
        """Calculate RFM on a given DataFrame

        Arguments:
            df {DataFrame} -- DataFrame where to calculate RFM
            reference_date {date} -- Date for recency calculation

        Returns:
            DataFrame -- Data Frame with Recency, Frequency and Monetary and scores
        """
        RFM = df.groupby('customer_unique_id').agg(
            Recency=pd.NamedAgg(
                column='order_purchase_timestamp',
                aggfunc=lambda x: (reference_date - x.max()).days),
            Frequency=pd.NamedAgg(
                column='customer_unique_id',
                aggfunc='count'),
            Monetary=pd.NamedAgg(
                column='payment_value',
                aggfunc='sum')
        )

        # RFM score
        quantiles = RFM.quantile(q=[0.25, 0.5, 0.75])
        quantiles = quantiles.to_dict()

        RFM['R_score'] = RFM['Recency'].apply(
            self.R_Class, args=('Recency', quantiles))
        RFM['F_score'] = RFM['Frequency'].apply(
            self.FM_Class, args=('Frequency', quantiles))
        RFM['M_score'] = RFM['Monetary'].apply(
            self.FM_Class, args=('Monetary', quantiles))

        return RFM

    def get_period(self, df, start_date, end_date):
        """Perform KMeans clustering on a data frame

        Arguments:
            df {DataFrame} -- [Data frame where to extract data]
            start_date {date} -- [min purchase order date]
            end_date {date} -- [max purchase order date]

        Returns:
            DataFrame -- Data Frame with RFM and clusters
        """
        df = df[(df['order_purchase_timestamp'] >= start_date)
                & (df['order_purchase_timestamp'] <= end_date)]
    
        # RFM calculation
        RFM = self.get_RFM(df, end_date)
        df = df.merge(
            RFM,
            left_on='customer_unique_id',
            right_on='customer_unique_id',
            how='left'
        )

        df.drop(columns=['R_score','F_score','M_score'], inplace=True)

        # Group data by customer
        df = df.groupby('customer_unique_id').mean()

        # Selected features to get clusters       
        data = df[self.features]

        #data['customer_id'] = np.log(data['customer_id'] + 0.001)
        try:
            data['Monetary'] = np.log(data['Monetary'] + 0.001)
        except:
            pass

        # Normalize data
        data_scaled = self.scaler.fit_transform(data)

        # Cluster with KMeans
        kmeans = KMeans(n_clusters=self.nb_cluster,
                        init='k-means++', random_state=2611)
        kmeans.fit(data_scaled)
        labels = kmeans.predict(data_scaled)

        # Add clusters to data set
        df['cluster'] = pd.Series(labels, index=data.index).astype(int)

        return df, kmeans



    def get_data(self, df, start_date, end_date):
        """Perform KMeans clustering on a data frame

        Arguments:
            df {DataFrame} -- [Data frame where to extract data]
            start_date {date} -- [min purchase order date]
            end_date {date} -- [max purchase order date]

        Returns:
            DataFrame -- Data Frame with RFM and clusters
        """
        df = df[(df['order_purchase_timestamp'] >= start_date)
                & (df['order_purchase_timestamp'] <= end_date)]
    
        # RFM calculation
        RFM = self.get_RFM(df, end_date)
        df = df.merge(
            RFM,
            left_on='customer_unique_id',
            right_on='customer_unique_id',
            how='left'
        )

        # Group data by customer
        df = self.get_customer_data(df)

        # Selected features to get clusters       
        data = df[self.features]

        try:
            data['Monetary'] = np.log(data['Monetary'] + 0.001)
        except:
            pass

        # Normalize data
        data_scaled = self.scaler.fit_transform(data)

        # Cluster with KMeans
        kmeans = KMeans(n_clusters=self.nb_cluster,
                        init='k-means++', random_state=2611)
        kmeans.fit(data_scaled)
        labels = kmeans.predict(data_scaled)

        # Add clusters to data set
        df['cluster'] = pd.Series(labels, index=data.index).astype(int)

        return df, kmeans

    def get_scores(self, df):
        """Calculate ARI and AMI score on DataFrame.
        The score is calculated between colums c0 (ground true) and cluster (pred) 

        Arguments:
            df {DataFrame} -- Data Frame

        Returns:
            list -- ARI score and AMI score
        """
        ari = metrics.adjusted_rand_score(
            df['c0'].tolist(), df['cluster'].tolist())
        ami = metrics.adjusted_mutual_info_score(
            df['c0'].tolist(), df['cluster'].tolist())

        return ari, ami

    def get_customer_data(self, df):
        """Group incoming DataFrame with customer_unique_id

        Arguments:
            df {DataFrame} -- DataFrame

        Returns:
            DataFrame -- DataFrame
        """
        customer_data = df.groupby('customer_unique_id').agg(
            {                   
                'order_id': 'count',
                'customer_id': 'count',
                'order_status': lambda x: x.mode()[0],
                'order_purchase_timestamp': 'max',
                'order_approved_at': 'max',
                'order_delivered_carrier_date': 'max',
                'order_delivered_customer_date': 'max',
                'order_estimated_delivery_date': 'max',
                'order_purchase_approved_minutes': 'mean',
                'order_purchase_carrier_days': 'mean',
                'order_purchase_delivered_days': 'mean',
                'order_carrier_delivered_days': 'mean',
                'order_delivered_estimated_days': 'mean',
                'order_item_id': 'count',
                'product_id': lambda x: x.mode()[0],
                'seller_id': lambda x: x.mode()[0],
                'shipping_limit_date': 'max',
                'price': 'mean',
                'freight_value': 'mean',
                'product_name_lenght': 'mean',
                'product_description_lenght': 'mean',
                'product_photos_qty': 'mean',
                'product_weight_g': 'mean',
                'product_length_cm': 'mean',
                'product_height_cm': 'mean',
                'product_width_cm': 'mean',
                'product_volume': 'mean',
                'seller_zip_code_prefix': lambda x: x.mode()[0],
                'seller_city': lambda x: x.mode()[0],
                'seller_state': lambda x: x.mode()[0],
                'payment_sequential': 'mean',
                'payment_type': lambda x: x.mode()[0],
                'payment_installments': 'mean',
                'payment_value': 'mean',
                'review_score': 'mean',
                'review_creation_date': 'max',
                'review_answer_timestamp': 'max',
                'review_have_title': 'mean',
                'review_have_message': 'mean',
                'review_creation_answer_days': 'mean',
                'customer_zip_code_prefix': lambda x: x.mode()[0],
                'customer_city': lambda x: x.mode()[0],
                'customer_state': lambda x: x.mode()[0],
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'sum'
            }
        ).reset_index()
        return customer_data


    def merge_periods(self, ref_period, period):
        """Merge to periods

        Arguments:
            ref_period {DataFrame} -- Base DataFrame 
            period {DataFrame} -- Period DataFrame

        Returns:
            DataFrame -- DataFrame
        """
        period = period.merge(
            ref_period,
            on='customer_unique_id',            
            how='outer'
        )
        # Fill NaN with default identified cluster
        period['c0'] = period['c0'].fillna(value=self.default_cluster)
        return period

    def merge_dataset(self):
        """Merge all dataset and add columns
        """

        # geolocation
        geolocation = pd.read_csv(
            self.base_directory + "olist_geolocation_dataset.csv")
        geolocation = geolocation.drop_duplicates(
            subset=['geolocation_zip_code_prefix'], keep="first")

        # Sellers
        sellers = pd.read_csv(self.base_directory +
                              "olist_sellers_dataset.csv")
        sellers = sellers.merge(
            geolocation[['geolocation_lat', 'geolocation_lng',
                         'geolocation_zip_code_prefix']],
            left_on='seller_zip_code_prefix',
            right_on='geolocation_zip_code_prefix',
            how='left'
        )
        sellers.drop('geolocation_zip_code_prefix', axis=1, inplace=True)
        sellers.rename(columns={"geolocation_lat": "seller_geolocation_lat",
                                "geolocation_lng": "seller_geolocation_lng"}, inplace=True)

        # Products
        products = pd.read_csv(self.base_directory +
                               "olist_products_dataset.csv")
        products['product_volume'] = products['product_length_cm'] * \
            products['product_height_cm'] * products['product_width_cm']

        # order items
        order_items = pd.read_csv(
            self.base_directory + "olist_order_items_dataset.csv")
        order_items = order_items.merge(
            products, left_on='product_id', right_on='product_id', how='left')
        order_items = order_items.merge(
            sellers, left_on='seller_id', right_on='seller_id', how='left')

        # Customers
        customers = pd.read_csv(self.base_directory +
                                "olist_customers_dataset.csv")
        customers = customers.merge(geolocation[['geolocation_lat', 'geolocation_lng', 'geolocation_zip_code_prefix']],
                                    left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
        customers.drop('geolocation_zip_code_prefix', axis=1, inplace=True)
        customers.rename(columns={"geolocation_lat": "customer_geolocation_lat",
                                  "geolocation_lng": "customer_geolocation_lng"}, inplace=True)

        # Review
        order_reviews = pd.read_csv(
            self.base_directory + "olist_order_reviews_dataset.csv")
        order_reviews['review_answer_timestamp'] = pd.to_datetime(
            order_reviews['review_answer_timestamp'])
        order_reviews['review_creation_date'] = pd.to_datetime(
            order_reviews['review_creation_date'])


        order_reviews['review_have_title'] = order_reviews.apply(
            lambda x: 0 if pd.isnull(x['review_comment_title']) else 1, axis=1)

        order_reviews['review_have_message'] = order_reviews.apply(
            lambda x: 0 if pd.isnull(x['review_comment_message']) else 1, axis=1)

        order_reviews['review_creation_answer_days'] = (
            order_reviews['review_answer_timestamp'] - order_reviews['review_creation_date']).dt.days

        # Payment
        order_payments = pd.read_csv(
            self.base_directory + "olist_order_payments_dataset.csv")

        # Orders
        orders = pd.read_csv(self.base_directory + "olist_orders_dataset.csv")

        # Leadtime in minutes between purchase and order approvement
        orders['order_approved_at'] = pd.to_datetime(
            orders['order_approved_at'])
        orders['order_purchase_timestamp'] = pd.to_datetime(
            orders['order_purchase_timestamp'])
        orders['order_purchase_approved_minutes'] = (
            orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.total_seconds().div(60)
        
        # Leadtime in minutes between purchase and delivery to carrier
        orders['order_delivered_carrier_date'] = pd.to_datetime(
            orders['order_delivered_carrier_date'])
        orders['order_purchase_carrier_days'] = (
            orders['order_delivered_carrier_date'] - orders['order_purchase_timestamp']).dt.days

        # Null values (not yet delivered to carrier) are replaced by NaN
        orders.loc[orders['order_purchase_carrier_days'] < 0] = np.NaN

        # Leadtime in days between purchase and delivery to customer
        orders['order_delivered_customer_date'] = pd.to_datetime(
            orders['order_delivered_customer_date'])
        orders['order_purchase_delivered_days'] = (
            orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days

        # Leadtime in days between carrier delivery and customer delivery
        orders['order_carrier_delivered_days'] = (
            orders['order_delivered_customer_date'] - orders['order_delivered_carrier_date']).dt.days

        # Leadtime in days between estimated delivery and customer delivery
        orders['order_estimated_delivery_date'] = pd.to_datetime(
            orders['order_estimated_delivery_date'])
        orders['order_delivered_estimated_days'] = (
            orders['order_estimated_delivery_date'] - orders['order_delivered_customer_date']).dt.days

        orders = orders.merge(order_items, left_on='order_id',
                              right_on='order_id', how='left')
        orders = orders.merge(
            order_payments, left_on='order_id', right_on='order_id', how='left')
        orders = orders.merge(
            order_reviews, left_on='order_id', right_on='order_id', how='left')
        orders = orders.merge(customers, left_on='customer_id',
                              right_on='customer_id', how='left')

        orders.dropna(subset=['order_id', 'customer_id',
                              'customer_unique_id'], inplace=True)
        return orders

    def use_kmeans(self, data):
        """Use KMeans model for clustering

        Arguments:
            data {DataFrame} -- Data Frame

        Keyword Arguments:
            preprocessing {str} -- Scaler for data normalisation (default: {'minmax'})
        """

        try:
            data['Monetary'] = np.log(data['Monetary'] + 0.001)
        except:
            pass
        
        data_scaled = self.scaler.fit_transform(data)
        SSE = []

        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=2611)
            kmeans.fit(data_scaled)
            labels = kmeans.predict(data_scaled)
            print(
                f'Silhouette Score(n={k}): {silhouette_score(data_scaled, labels)}')
            SSE.append(kmeans.inertia_)
        
        fig = plt.figure(figsize=(20, 10))
        title = fig.suptitle("Progression du score Silhouette", fontsize=18)
        sns.pointplot(x=list(range(2, 7)), y=SSE)
        plt.show()

    def use_DBSCAN(self, data, eps=0.2, min_samples=100):
        """Use DBSCAN for clustering

        Arguments:
            data {DataFrame} -- DataFrame

        Keyword Arguments:
            eps {float} -- Epsilon (default: {0.2})
            min_samples {int} -- Min of sample (default: {100})

        Returns:
            list -- Labels
        """        
        try:
            data['Monetary'] = np.log(data['Monetary'] + 0.001)
        except:
            pass

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
        labels = db.labels_
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print(f'Estimation du nombre de clusters: {n_clusters_}')
        print(f'Estimation du nombre de points considrés comme du bruit: {n_noise_}')
        
        print(
                f'Silhouette Score(n={n_clusters_}): {silhouette_score(data_scaled, labels)}')
                       
        return labels

    def display_parallel_coordinates_centroids(self, df, image_name):
        """Display a parallel coordinates plot for the centroids in df.
            A columns named 'cluster' is needed
        Arguments:
            df {dataframe} -- Dataframe to display
            image_name {string} -- Filename to save the plots
        """        
        # Create the plot
        fig = plt.figure(figsize=(20, 10))
        title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
        fig.subplots_adjust(top=0.9, wspace=0)

        # Draw the chart
        parallel_coordinates(df, 'cluster', color=palette)

        # Stagger the axes
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20) 
        
        plt.savefig('img/' + image_name + '.png')                 
        plt.show()

    
    def display_kmeans_centroids(self, data, image_name='none'):
        """Display centroids of kmeans models

        Arguments:
            data {DataFrame} -- DataFrame

        Keyword Arguments:
            image_name {str} -- filename for the saved image (default: {'none'})

        Returns:
            list -- Predicted labels
        """        
    
        try:
            data['Monetary'] = np.log(data['Monetary'] + 0.001)
        except:
            pass
        data_scaled = self.scaler.fit_transform(data)
    
        kmeans = KMeans(n_clusters=self.nb_cluster, random_state=2611)
        kmeans.fit(data_scaled)
        labels = kmeans.predict(data_scaled)
        palette = sns.color_palette("bright", 4)
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=data.columns)
        centroids['cluster'] = centroids.index
        self.display_parallel_coordinates_centroids(centroids, image_name)
    
    
        return labels


    def display_customers(self, data, labels, col):
        """Display clusters by % of customer and by % of revenue

        Arguments:
            data {DataFrame} -- Data Frame
            labels {List} -- Cluster list
            col {string} -- Columns for sum
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))
        

        monetary_sum = data[col].sum()
        clusters = data.groupby('cluster').agg({
            col: 'sum'
        })    
        col_percent = clusters.groupby(level=0).apply(lambda x: 100 * x / monetary_sum)
        
        ax1.set_title("Pourcentage du chiffre d'affaire au sein des groupes")
        ax1.pie(col_percent,
            autopct='%1.1f%%',
            shadow=True,
            startangle=180,
            labels=labels
        )

        clusters = data.groupby('cluster').mean()        
        ax2.set_title("Pourcentage des clients au sein des groupes")
        ax2.pie(clusters['review_score'],
            autopct='%1.1f%%',
            shadow=True,
            startangle=180,
            labels=labels
        )
        plt.show()


    def display_dbscan_core_points(self, data,  labels, image_name='none'):
        """Display centoid (aka core point) of DBScan model

        Arguments:
            data {DataFrame} -- Data Frame
            labels {list} -- List of cluster

        Keyword Arguments:
            image_name {str} -- Filename to save as (default: {'none'})
        """        
        try:
            data['Monetary'] = np.log(data['Monetary'] + 0.001)
        except:
            pass
        data_scaled = self.scaler.fit_transform(data)
    
        # Center points
        core_points = []
        for i in range(self.nb_cluster):
            core_points.append(np.mean(data_scaled[labels==i,:], axis=0))

        palette = sns.color_palette("bright", 4)
        centroids = pd.DataFrame(core_points, columns=data.columns)
        centroids['cluster'] = range(self.nb_cluster)
        self.display_parallel_coordinates_centroids(centroids, image_name)

    def use_CAH(self, data): 
        """Use CAH Model

        Arguments:
            data {DataFrame} -- DataFrame

        Returns:
            array -- Linkage matrix
        """      
        try:
            data['Monetary'] = np.log(data['Monetary'] + 0.001)
        except:
            pass  
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
    
        Z = linkage(data_scaled, method='ward', metric='euclidean')
        plt.title("CAH")
        dendrogram(Z, labels=data.index, orientation='left', color_threshold=0)
        plt.show() 
    
        return Z

    def display_CAH(self, data, Z):
        """Graphic representation of CAH

        Arguments:
            data {DataFrame} -- DataFrame
            Z {array} -- Linkage matrix

        Returns:
            list -- Cluster list
        """        
        data_scaled = self.scaler.fit_transform(data)
    
        groupes_cah = fcluster(Z, t=self.nb_cluster, criterion='distance')
        idg = np.argsort(groupes_cah)
        print(f'Silhouette Score(n={3}): {silhouette_score(data_scaled, groupes_cah)}')
        return groupes_cah

    def get_by_period(self, orders, features, purchase_end_date, months):
        purchase_start_date = purchase_end_date + relativedelta(months=months)
        period = orders[(orders['order_purchase_timestamp'] >= purchase_start_date) & (orders['order_purchase_timestamp'] <= purchase_end_date)]
    
        # RFM calculation
        RFM = self.get_RFM(period, purchase_end_date)
        period = period.merge(RFM, left_on='customer_unique_id',right_on='customer_unique_id',how='left')

        # Group data by customer
        period = self.get_customer_data(period)
        period = period[features]

        #self.use_kmeans(period)

        return period

