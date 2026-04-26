from django.contrib import admin

from .models import (
    Address,
    Coupon,
    FraudAlert,
    FraudBlocklist,
    FraudModelReport,
    FraudPolicy,
    FraudPolicyAuditLog,
    FraudTransaction,
    Item,
    Order,
    OrderItem,
    Payment,
    Refund,
    UserProfile,
)


def make_refund_accepted(modeladmin, request, queryset):
    queryset.update(refund_requested=False, refund_granted=True)


make_refund_accepted.short_description = 'Update orders to refund granted'


class OrderAdmin(admin.ModelAdmin):
    list_display = ['user',
                    'ordered',
                    'being_delivered',
                    'received',
                    'refund_requested',
                    'refund_granted',
                    'shipping_address',
                    'billing_address',
                    'payment',
                    'coupon'
                    ]
    list_display_links = [
        'user',
        'shipping_address',
        'billing_address',
        'payment',
        'coupon'
    ]
    list_filter = ['ordered',
                   'being_delivered',
                   'received',
                   'refund_requested',
                   'refund_granted']
    search_fields = [
        'user__username',
        'ref_code'
    ]
    actions = [make_refund_accepted]


class AddressAdmin(admin.ModelAdmin):
    list_display = [
        'user',
        'street_address',
        'apartment_address',
        'country',
        'zip',
        'address_type',
        'default'
    ]
    list_filter = ['default', 'address_type', 'country']
    search_fields = ['user', 'street_address', 'apartment_address', 'zip']


def mark_alerts_closed(modeladmin, request, queryset):
    queryset.update(status='closed')


mark_alerts_closed.short_description = 'Đánh dấu cảnh báo đã đóng'


@admin.register(FraudPolicy)
class FraudPolicyAdmin(admin.ModelAdmin):
    list_display = [
        'name',
        'is_active',
        'fraud_threshold',
        'block_transaction',
        'ban_user',
        'block_card_fingerprint',
        'manual_review',
        'send_email_alert',
        'updated_at',
    ]
    list_filter = [
        'is_active',
        'block_transaction',
        'ban_user',
        'block_card_fingerprint',
        'manual_review',
        'send_email_alert',
    ]
    search_fields = ['name']


@admin.register(FraudBlocklist)
class FraudBlocklistAdmin(admin.ModelAdmin):
    list_display = ['block_type', 'block_value', 'is_active', 'reason', 'updated_at']
    list_filter = ['block_type', 'is_active']
    search_fields = ['block_value', 'reason']


@admin.register(FraudTransaction)
class FraudTransactionAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'external_transaction_id',
        'external_user_id',
        'amount',
        'fraud_score',
        'is_fraud_prediction',
        'decision',
        'model_name',
        'model_version',
        'created_at',
    ]
    list_filter = ['decision', 'is_fraud_prediction', 'currency', 'model_name']
    search_fields = ['external_transaction_id', 'external_user_id', 'card_fingerprint']
    ordering = ['-created_at']


@admin.register(FraudAlert)
class FraudAlertAdmin(admin.ModelAdmin):
    list_display = ['id', 'transaction', 'severity', 'status', 'created_at']
    list_filter = ['severity', 'status']
    search_fields = ['message', 'transaction__external_transaction_id']
    actions = [mark_alerts_closed]
    ordering = ['-created_at']


@admin.register(FraudModelReport)
class FraudModelReportAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'model_version', 'is_champion', 'created_at']
    list_filter = ['is_champion', 'model_name']
    search_fields = ['model_name', 'model_version']
    ordering = ['-created_at']


@admin.register(FraudPolicyAuditLog)
class FraudPolicyAuditLogAdmin(admin.ModelAdmin):
    list_display = ['policy', 'change_source', 'changed_by', 'created_at']
    list_filter = ['change_source', 'policy']
    search_fields = ['payload']
    ordering = ['-created_at']


admin.site.register(Item)
admin.site.register(OrderItem)
admin.site.register(Order, OrderAdmin)
admin.site.register(Payment)
admin.site.register(Coupon)
admin.site.register(Refund)
admin.site.register(Address, AddressAdmin)
admin.site.register(UserProfile)
