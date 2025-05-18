import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# SECURITY WARNING: Modify this secret key if using in production!
SECRET_KEY = "6few3nci_q_o@l1dlbk81%wcxe!*6r29yu629&d97!hiqat9fa"

DEFAULT_AUTO_FIELD='django.db.models.AutoField'

DATABASES = {
    "default": {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME':  'KarelDB', #'DataPiramide DataSPDev DataSP, DataSPValidation, dataMonitorTJSP2', #"dataTJCE", DataTraders, DataRCX, DataTradersAssociate
        'USER': 'rubens',
        'PASSWORD': '63632323',
        'HOST': '200.235.131.142',
        'PORT': '5432',
    }
}

INSTALLED_APPS = ("data",)
