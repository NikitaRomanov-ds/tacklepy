

'''The provided settings are parameter configurations for different machine learning models used in the DataImputor module
for predicting missing values in data. They include settings for HistGradientBoosting, XGBoost, and CatBoost models,
covering binary classification, multi-class classification, and regression tasks. These settings define the specific
parameters and configurations for each model, enabling accurate prediction and imputation of missing values in the data.
'''

binary_classification_params_hgb = {'loss':'log_loss',
                                    'learning_rate':0.1,
                                    'max_iter':100,
                                    'max_depth':3,
                                    'l2_regularization':0.1,
                                    'max_bins':255}


multiclass_classification_params_hgb = {'loss':'log_loss',
                                        'learning_rate':0.1,
                                        'max_iter':100,
                                        'max_depth':3,
                                        'l2_regularization':0.1,
                                        'max_bins':255}

regression_params_hgb = {'loss':'squared_error',
                         'learning_rate':0.1,
                         'max_iter':100,
                         'max_depth':3,
                         'l2_regularization':0.1,
                         'max_bins':255}


binary_classification_parameters_xgb = {'objective':'binary:logistic',
                                        'use_label_encoder':False,
                                        'eval_metric':'logloss',
                                        'learning_rate':0.1,
                                        'n_estimators':100,
                                        'max_depth':3,
                                        'subsample':0.9, 
                                        'colsample_bytree':0.7, 
                                        'reg_alpha':0.01, 
                                        'reg_lambda':0.01, 
                                        'n_jobs':-1}

multi_classification_parameters_xgb = {'objective':'multi:softmax',
                                        'use_label_encoder':False,
                                        'eval_metric':'mlogloss',
                                        'learning_rate':0.1,
                                        'n_estimators':100, 
                                        'max_depth':3,
                                        'subsample':0.9,
                                        'colsample_bytree':0.7,
                                        'reg_alpha':0.01,
                                        'reg_lambda':0.01,
                                        'n_jobs':-1}

regression_parameters_xgb = {'objective':'reg:squarederror',
                            'learning_rate':0.1,
                            'n_estimators':100, 
                            'max_depth':3,
                            'subsample':0.9,
                            'colsample_bytree':0.7,
                            'reg_alpha':0.01,
                            'reg_lambda':0.01,
                            'n_jobs':-1}

binary_classification_parameters_catboost = {'loss_function': 'Logloss',
                                             'eval_metric': 'Accuracy',
                                             'learning_rate': 0.1,
                                             'iterations': 100,
                                             'depth': 5,
                                             'subsample': 0.9,
                                             'colsample_bylevel': 0.7,
                                             'l2_leaf_reg': 0.01,
                                             'thread_count': -1}

multi_classification_parameters_catboost = {'loss_function': 'MultiClass',
                                            'eval_metric': 'MultiClass',
                                            'learning_rate': 0.1,
                                            'iterations': 100,
                                            'depth': 5,
                                            'early_stopping_rounds': 10,
                                            'thread_count': -1}

regression_parameters_catboost = {'loss_function': 'RMSE',
                                  'eval_metric': 'RMSE',
                                  'learning_rate': 0.1,
                                  'iterations': 100,
                                  'depth': 5,
                                  'early_stopping_rounds': 10,
                                  'thread_count': -1}




class DataImputer:

    """
    The DataImputer class handles missing values in a dataset by predicting and imputing those missing values.
    It provides functionalities for imputing numerical and categorical columns separately. The class uses various machine
    learning algorithms such as HistGradientBoosting, XGBoost, or CatBoost to predict missing values based on highly
    correlated features. The algorithm used for imputation can be selected by the user. The class also handles outliers
    in the data before performing imputation. It supports:

    - binary classification

    - multi-class classification

    - regression tasks

    Depending on the type of column being imputed. The class provides options to exclude specific
    columns from imputation, control verbosity, and define the size of the training set.

    The DataImputer class aims to automate the process of handling missing values and enhance the completeness of the dataset.


        PARAMETERS:

        - exclude (default None). You can set a list of columns to be expluded from all the processes.

        - verbose (default True). When it is set to True, you get the full information about running processes at the moment.
                     When set to False, none information about execution phases is returned.

        - algorithm default 'xgb'). You can choose and set any of the following model libraries to predict NaN values:

                    'xgb' for XGBoost Classifier and Regressor

                    'hgb' for HistGradientBoosting Classifier and Regressor

                    'cat' for CatBoost Classifier and Regressor

        - train_size (default 'all'). You can choose size of model training data. You can set it to default - all not null data.
                     You can manually set size of data you prefer with any integer you like.

                     In case train_size is not default, please specify:

                     - random_state (default None) set it with any digit you like. This ensures that the data is split consistently
                     across different runs and provides reproducibility.

                     - shuffle (default False). The shuffle parameter, when set to True, enables shuffling of the
                     training data before splitting it. This can be useful to introduce randomness in the training process, especially
                     when the dataset has a specific order or structure. If shuffling is desired, set shuffle to True; otherwise, set
                     it to False to maintain the original order of the data during training.

            Please note that if the train_size parameter is set to the default value of 'all', the random_state and shuffle parameters
            are not relevant and can be left unspecified.

    """

    def __init__(self, exclude=None, verbose=True, algorithm='xgb', train_size='all', random_state=None, shuffle=False):
        self._exclude = exclude or []
        self.sorted_values = {}
        self.verbose = verbose
        self.algorithm = algorithm
        self.reverse_sorted_values = {}
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
    
    def _find_highly_correlated_features(self, data, col):

        '''
        The method identifies highly correlated features with a specific column in a given dataset. It excludes certain
        columns specified by self._exclude and those with data types 'object' or 'datetime'. This method calculates
        the correlations between the non-null values of the target column and other numeric columns in the dataset.
        It sorts the correlations in descending order, selects a maximum of 12 columns (or less if there are fewer
        correlations available), and returns the names of the highly correlated features.

        The purpose of this method is to identify relevant features that exhibit strong correlation with a specific column,
        which can be useful for predicting and imputing missing values in that column based on the values of the correlated features.
        '''

        exclude_cols = self._exclude + [col]
        unsupported_cols = data.drop(exclude_cols, axis=1).select_dtypes(include=['O', 'datetime']).columns.tolist()

        non_null_data = data.dropna(subset=[col])

        correlations = non_null_data.drop(exclude_cols + unsupported_cols, axis=1).apply(lambda x: x.corr(non_null_data[col]))
        correlations = correlations.abs().sort_values(ascending=False)

        num_cols_needed = min(12, len(correlations) + 1)  # to handle less number of columns in correlations
        high_corr_feats = list(correlations[:num_cols_needed].index)

        if col not in high_corr_feats:
            high_corr_feats.append(col)

        while len(high_corr_feats) < num_cols_needed:
            selected_data = data[high_corr_feats].dropna()

            X = selected_data.values

            vif = pd.DataFrame()
            vif["Features"] = high_corr_feats
            vif["VIF"] = [variance_inflation_factor(X, i) for i in range(len(high_corr_feats))]

            max_vif_idx = vif['VIF'].idxmax()
            max_vif = vif['VIF'].max()

            if np.isnan(max_vif) or max_vif <= 5:
                break

            min_corr_feature = vif['Features'].iloc[max_vif_idx]
            high_corr_feats.remove(min_corr_feature)

            max_corr_feature = None
            max_corr = -1
            max_corr_vif = np.inf

            for feature in correlations.drop(high_corr_feats).index:
                new_feats = high_corr_feats + [feature]
                selected_data = data[new_feats].dropna()

                X = selected_data.values
                vif = [variance_inflation_factor(X, i) for i in range(len(new_feats))]
                max_new_vif = max(vif)
                corr = correlations[feature]

                if corr > max_corr and max_new_vif <= 5 and max_new_vif < max_corr_vif:
                    max_corr_feature = feature
                    max_corr = corr
                    max_corr_vif = max_new_vif

            if max_corr_feature is None:
                break

            high_corr_feats.append(max_corr_feature)

        numeric_cols = data[high_corr_feats].select_dtypes(include=np.number).columns.tolist()
        self.sorted_values[col] = high_corr_feats
        return high_corr_feats


    def _find_nan_columns(self, data):

        '''
        This method scans dataset to identify columns that contain missing values (NaN).
        It excludes any columns specified in self._exclude. For each column, the method checks if it is of a
        numeric data type and counts the total number of values, the number of non-null values,
        and the number of null values. If the column has null values and the proportion of null values is
        less than or equal to 50% of the total values, it is considered for imputation. The method keeps track
        of the columns with null values and provides information about the number of total values, non-null values,
        and null values for each column if self.verbose is set to True.

        At the end of running, method returns a list of column names that have null values and satisfy the criteria for imputation.
        '''

        nan_columns_list = []
        sep_count = 0
        exclude_cols = set(self._exclude)

        for col in data:
            if col in exclude_cols:
                continue

            if pd.api.types.is_numeric_dtype(data[col]):
                total_count = data[col].shape[0]
                null_count = data[col].isna().sum()
                not_null_count = total_count - null_count

                if null_count > 0 and null_count <= total_count * 0.5:
                    if sep_count == 0 and self.verbose:
                        print("=" * terminal_width)
                        phrase = "\033[1mNaN IMPUTATION INFO:\n\033[0m"
                        centered_phrase = phrase.center(terminal_width)
                        print(centered_phrase)

                    if self.verbose:
                        print(f"{col}: Total={total_count}, Not null={not_null_count}, \033[1mNull={null_count}\033[0m")

                    sep_count += 1
                    nan_columns_list.append(col)

        if sep_count > 0 and self.verbose:
            print("=" * terminal_width)

        if self.verbose:
            phrase = f"\n\033[1mTotal columns with NaN values for prediction: {len(nan_columns_list)}\033[1m\n"
            centered_phrase = phrase.center(terminal_width)
            print(centered_phrase)
            print("=" * terminal_width)

        return nan_columns_list

    def _model_type_defination(self, col):

        '''
        Model settings according to the selected algorithm to predict NaN values. Here are steted parameters for all types of
        models available for Data Imputation.
        '''

        if col.nunique() == 2:
            if self.algorithm == 'hgb':
                model = HistGradientBoostingClassifier(max_leaf_nodes=2, **binary_classification_params_hgb)
            elif self.algorithm == 'xgb':
                model = XGBClassifier(num_class=2, **binary_classification_parameters_xgb)
            elif self.algorithm == 'cat':
                model = CatBoostClassifier(**binary_classification_parameters_catboost)
        elif 30 >= col.nunique() > 2:
            if self.algorithm == 'hgb':
                model = HistGradientBoostingClassifier(max_leaf_nodes=col.nunique(), **multiclass_classification_params_hgb)
            elif self.algorithm == 'xgb':
                model = XGBClassifier(num_class=col.nunique(), **multi_classification_parameters_xgb)
            elif self.algorithm == 'cat':
                model = CatBoostClassifier(**multi_classification_parameters_catboost)
        else:
            if self.algorithm == 'hgb':
                model = HistGradientBoostingRegressor(**regression_params_hgb)
            elif self.algorithm == 'xgb':
                model = XGBRegressor(**regression_parameters_xgb)
            elif self.algorithm == 'cat':
                model = CatBoostRegressor(**regression_parameters_catboost)
        return model


    def _get_cat_binary_transformation_dict(self, data):

        '''
        This method creates a dictionary that maps unique categorical values in the specified columns to numerical indices.
        It iterates over columns in the input data that have an object (string) data type and generates a dictionary for each column.
        Each unique value in the column is assigned a unique index. The method then returns a dictionary where the keys are the
        column names, and the values are the corresponding transformation dictionaries for each column.
        '''

        transformation_dict = {}
        for col in data.select_dtypes(include='O'):
            unique_values = data[col].dropna().unique()
            col_dict = {value: index for index, value in enumerate(unique_values)}
            transformation_dict[col] = col_dict
        return transformation_dict

    def _change_df_values(self, data, new_values):

        '''
        This method replaces values in the specified columns of a DataFrame with their corresponding values from a given dictionary.
        It iterates over the columns and their respective transformation dictionaries. For each column, it checks if it exists in
        the DataFrame and creates a reverse dictionary where the keys and values are swapped. It then uses this reverse dictionary
        to map the values in the column to their original values. If the column's data type is categorical, it converts the mapped
        values to categorical data type using the categories from the transformation dictionary. Any remaining missing values in the
        column are filled with the original values. The method returns the modified DataFrame with the updated column values.

        In general, this block of code is responsible for feature back transformation to the original feature view after NaN values
        were predicted and imputed.
        '''

        for col, values_dict in new_values.items():
            if col in data.columns:
                reverse_dict = {v: k for k, v in values_dict.items()}
                if data[col].dtype.name == 'category':
                    data[col] = pd.Categorical(data[col].map(reverse_dict), categories=values_dict.keys())
                else:
                    data[col] = data[col].map(reverse_dict).fillna(data[col])
        return data


    def handle_outliers(self, data):

        '''
        This block of code handles outliers in the given dataset. It identifies the outliers in each numeric column using the interquartile
        range (IQR) method. The code calculates the lower and upper bounds based on the first quartile (Q1) and third quartile (Q3) of the
        column's data. Any value below the lower bound or above the upper bound is considered an outlier. The code replaces the outlier values
        with NaN in the dataset and collects information about the outliers in a list of tuples. Finally, it returns the updated dataset with
        outliers replaced by NaN values and the list of tuples containing information about the outliers.

        !!!
        Please note that the code assumes the input dataset contains only numeric columns. Origin data outliers will be returned back after
        NaN imputation will be compleated. This is done for better NaN value prediction accuracy.
        !!!

        '''

        tuples_outliers = []
        for col in data.select_dtypes(include=np.number):
            column_data = data[col]

            Q1 = np.percentile(column_data, 25)
            Q3 = np.percentile(column_data, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 2.0 * IQR
            upper_bound = Q3 + 2.0 * IQR

            outlier_indices = np.where((column_data < lower_bound) | (column_data > upper_bound))[0]

            outlier_values = column_data.iloc[outlier_indices]

            data.loc[outlier_indices, col] = np.nan

            tuples_outliers.extend([(col, index, value) for index, value in zip(outlier_indices, outlier_values)])

        return data, tuples_outliers

    def impute_nan(self, data):

        '''
        This block of code performs the actual imputation of NaN (missing) values in the dataset. It follows several steps:

        1. The code creates a copy of the input data and stores the original data for reference.

        2. It initializes a ColumnEnumerator object to transform categorical columns and handle unique values.

        3. The fit_transform method of the ColumnEnumerator is called to transform the data.

        4. It identifies columns with NaN values using the find_nan_columns method and stores them in nan_columns_list.

        5. Outliers are handled by calling the handle_outliers method, which replaces outlier values with NaN in the
        numeric columns and collects information about the outliers.

        6. For each column in nan_columns_list, the code predicts the missing values using highly correlated features.
        The highly correlated features are determined using the _find_highly_correlated_features method.

        7. The appropriate model is selected based on the column's characteristics using the _model_type_defination method.
        The missing values are predicted using the selected model, and the predictions are assigned to the NaN values in the dataset.

        8. The transformation dictionary is reversed and applied to the categorical columns in the data. If there are any columns
        that are completely filled with NaN values, the original values from the original data are retained.

        9. The numeric columns are transformed back using the change_df_values method.

        10. Outliers that were once represented as NaN values are restored in the data.

        11. The elapsed time of the imputation process is calculated, and if verbosity is enabled, it is printed along with a completion message.

        12. Garbage collection is performed to free up memory.

        13. The imputed data is returned.
        '''

        start_time = time.time()
        data = data.copy(deep=True)
        orig_data = data.copy(deep=True)

        column_enumerator = ColumnEnumerator()
        numeric_df, enumeration_dict, tuples_once_represented_values = column_enumerator.fit_transform(data)
        enumerated_column_list = list(enumeration_dict.keys())

        transformation_dict = self._get_cat_binary_transformation_dict(data)
        nan_columns_list = self._find_nan_columns(numeric_df)

        numeric_df, tuples_outliers = self.handle_outliers(numeric_df)

        for col in nan_columns_list:
            if col not in orig_data.columns:
                continue
            
            high_corr_feats = self._find_highly_correlated_features(numeric_df, col)
            high_corr_feats_excl_target = high_corr_feats.copy()
            high_corr_feats_excl_target.remove(col)
            if self.verbose:
                phrase = f"\n\033[1mNaN values prediction in column '{col}' using highly correlated features:\033[0m \n\n {high_corr_feats_excl_target}\n"
                centered_phrase = phrase.center(terminal_width)
                print(centered_phrase)

            model = self._model_type_defination(numeric_df[col])
            nan_ix = numeric_df[numeric_df[col].isna()].index
            test = numeric_df.loc[nan_ix, high_corr_feats].drop(col, axis=1)
            train = numeric_df.loc[~numeric_df[col].isna(), high_corr_feats]

            if self.train_size != 'all':
                train, _, _, _ = train_test_split(train, train[col], train_size=self.train_size, random_state=self.random_state, shuffle=self.shuffle)

            X_train = train.drop(col, axis=1)
            y_train = train[col]

            if self.algorithm == 'hgb':
                model.fit(X_train, y_train)
                pred = model.predict(test)
            elif self.algorithm == 'xgb':
                model.fit(X_train, y_train, verbose=False)
                pred = model.predict(test)
            else:
                model.fit(X_train, y_train, verbose=False)
                pred = model.predict(test)

            if y_train.apply(lambda x: str(x).endswith('.0')).all():
                pred = np.round(pred)

            numeric_df.loc[nan_ix, col] = pred
            if self.verbose:
                phrase = f"All NaN values for column \033[1m'{col}'\033[0m are imputed\n"
                print(phrase.center(terminal_width))
                print("=" * terminal_width)

        for key in transformation_dict.keys():
            transformation_dict[key] = dict(map(reversed, transformation_dict[key].items()))
            data[key] = data[key].map(transformation_dict[key])

        nan_cols = data.columns[data.isna().all()]
        if len(nan_cols) > 0:
            data[nan_cols] = orig_data[nan_cols]

        numeric_df = self._change_df_values(numeric_df, enumeration_dict)
        data[numeric_df.columns] = numeric_df

        for col, index, value in tuples_outliers:
            tuples_once_represented_values.append((col, index, value))

        for col, index, value in tuples_once_represented_values:
            data.loc[index, col] = value

        for col in numeric_df.columns:
            if col in orig_data.columns and pd.api.types.is_numeric_dtype(orig_data[col]):
                if not (orig_data[col] < 0).any():
                    data[col] = np.where((data[col] < 0) | (data[col] == -0.0), np.abs(data[col]), data[col])

        elapsed_time = time.time() - start_time

        if self.verbose:
            elapsed_seconds = int(elapsed_time % 60)
            elapsed_minutes = int((elapsed_time // 60) % 60)
            elapsed_hours = int((elapsed_time // 3600) % 24)
            elapsed_days = int(elapsed_time // 86400)
            elapsed_fractional = elapsed_time % 1

            time_components = []
            if elapsed_days > 0:
                time_components.append(f"{elapsed_days} days")
            if elapsed_hours > 0:
                time_components.append(f"{elapsed_hours} hours")
            if elapsed_minutes > 0:
                time_components.append(f"{elapsed_minutes} minutes")
            time_components.append(f"{elapsed_seconds + elapsed_fractional:.2f} seconds")

            elapsed_time_str = ", ".join(time_components)

            print(f"\033[1mALL FEASIBLE NaNs IN DATAFRAME ARE IMPUTED\033[0m\n")
            print(f"\033[1mElapsed time: {elapsed_time_str}\033[0m")

        gc.collect()

        return data