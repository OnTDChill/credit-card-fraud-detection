from django.urls import path
from .fraud_views import (
    fraud_batch_score_view,
    fraud_ingest_view,
    fraud_metrics_view,
    fraud_policy_view,
    fraud_stream_view,
)
from .views import (
    ItemDetailView,
    CheckoutView,
    HomeView,
    OrderSummaryView,
    add_to_cart,
    remove_from_cart,
    remove_single_item_from_cart,
    PaymentView,
    AddCouponView,
    RequestRefundView
)

app_name = 'core'

urlpatterns = [
    path('', HomeView.as_view(), name='home'),
    path('checkout/', CheckoutView.as_view(), name='checkout'),
    path('order-summary/', OrderSummaryView.as_view(), name='order-summary'),
    path('product/<slug>/', ItemDetailView.as_view(), name='product'),
    path('add-to-cart/<slug>/', add_to_cart, name='add-to-cart'),
    path('add-coupon/', AddCouponView.as_view(), name='add-coupon'),
    path('remove-from-cart/<slug>/', remove_from_cart, name='remove-from-cart'),
    path('remove-item-from-cart/<slug>/', remove_single_item_from_cart,
         name='remove-single-item-from-cart'),
    path('payment/<payment_option>/', PaymentView.as_view(), name='payment'),
    path('request-refund/', RequestRefundView.as_view(), name='request-refund'),
    path('api/fraud/ingest/', fraud_ingest_view, name='fraud-ingest'),
    path('api/fraud/batch-score/', fraud_batch_score_view, name='fraud-batch-score'),
    path('api/fraud/stream/', fraud_stream_view, name='fraud-stream'),
    path('api/fraud/metrics/', fraud_metrics_view, name='fraud-metrics'),
    path('api/fraud/policy/', fraud_policy_view, name='fraud-policy'),
]
