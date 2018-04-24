"""stock_predict_show URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from . import view

url_prefix = 'show/'
urlpatterns = [
    url(r'^' + url_prefix + '$', view.index),
    url(r'^' +  url_prefix + 'cluster', view.cluster, name='cluster'),
    url(r'^' +  url_prefix + 'predict', view.predict, name='predict'),
    url(r'^' +  url_prefix + 'method', view.method, name='method'),
    url(r'^' +  url_prefix + 'action/get_cluster_result', view.get_cluster_result, name='get_cluster_result'),
    url(r'^' +  url_prefix + 'action/get_predict_result', view.get_predict_result, name='get_predict_result'),
]
