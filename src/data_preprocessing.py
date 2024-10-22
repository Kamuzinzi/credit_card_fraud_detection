import logging
logging.basicConfig(level=logging.INFO)

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTETomek



def handle_imbalance(X_train, y_train, method='none'):
    logging.info(f"Method selected: {method}")
    if method == 'SMOTE':
        # SMOTE (Synthetic Minority Over-sampling Techinique)
        smote = SMOTE()
        return smote.fit_resample(X_train, y_train)
    elif method == 'Undersampling':
        # Random Undersampling
        rus = RandomUnderSampler(random_state=101, sampling_strategy=0.8)
        return rus.fit_resample(X_train, y_train)
    elif method == 'Oversampling':
        print(method)
        # Random Oversampling
        ros = RandomOverSampler(random_state=111)
        return ros.fit_resample(X_train, y_train)
    elif method == 'Tomek':
        #Tomek_undersampling
        tomek = TomekLinks()
        return tomek.fit_resample(X_train, y_train)
    elif method == 'SMOTETomek':
        # Combine SMOTE and Tomek Links
        smote_tomek = SMOTETomek(random_state=20)
        return smote_tomek.fit_resample(X_train, y_train)
    elif method == 'ADASYN':
        # ADASYN oversampling
        adasyn = ADASYN(random_state=211)
        return adasyn.fit_resample(X_train, y_train)
    return X_train, y_train  # Return original if no method is applied