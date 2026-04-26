from django.db.models.signals import post_save
from django.conf import settings
from django.db import models
from django.db.models import Sum
from django.shortcuts import reverse
from django_countries.fields import CountryField


CATEGORY_CHOICES = (
    ('S', 'Shirt'),
    ('SW', 'Sport wear'),
    ('OW', 'Outwear')
)

LABEL_CHOICES = (
    ('P', 'primary'),
    ('S', 'secondary'),
    ('D', 'danger')
)

ADDRESS_CHOICES = (
    ('B', 'Billing'),
    ('S', 'Shipping'),
)


class UserProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    stripe_customer_id = models.CharField(max_length=50, blank=True, null=True)
    one_click_purchasing = models.BooleanField(default=False)

    def __str__(self):
        return self.user.username


class Item(models.Model):
    title = models.CharField(max_length=100)
    price = models.FloatField()
    discount_price = models.FloatField(blank=True, null=True)
    category = models.CharField(choices=CATEGORY_CHOICES, max_length=2)
    label = models.CharField(choices=LABEL_CHOICES, max_length=1)
    slug = models.SlugField()
    description = models.TextField()
    image = models.ImageField()

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse("core:product", kwargs={
            'slug': self.slug
        })

    def get_add_to_cart_url(self):
        return reverse("core:add-to-cart", kwargs={
            'slug': self.slug
        })

    def get_remove_from_cart_url(self):
        return reverse("core:remove-from-cart", kwargs={
            'slug': self.slug
        })


class OrderItem(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE)
    ordered = models.BooleanField(default=False)
    item = models.ForeignKey(Item, on_delete=models.CASCADE)
    quantity = models.IntegerField(default=1)

    def __str__(self):
        return f"{self.quantity} of {self.item.title}"

    def get_total_item_price(self):
        return self.quantity * self.item.price

    def get_total_discount_item_price(self):
        return self.quantity * self.item.discount_price

    def get_amount_saved(self):
        return self.get_total_item_price() - self.get_total_discount_item_price()

    def get_final_price(self):
        if self.item.discount_price:
            return self.get_total_discount_item_price()
        return self.get_total_item_price()


class Order(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE)
    ref_code = models.CharField(max_length=20, blank=True, null=True)
    items = models.ManyToManyField(OrderItem)
    start_date = models.DateTimeField(auto_now_add=True)
    ordered_date = models.DateTimeField()
    ordered = models.BooleanField(default=False)
    shipping_address = models.ForeignKey(
        'Address', related_name='shipping_address', on_delete=models.SET_NULL, blank=True, null=True)
    billing_address = models.ForeignKey(
        'Address', related_name='billing_address', on_delete=models.SET_NULL, blank=True, null=True)
    payment = models.ForeignKey(
        'Payment', on_delete=models.SET_NULL, blank=True, null=True)
    coupon = models.ForeignKey(
        'Coupon', on_delete=models.SET_NULL, blank=True, null=True)
    being_delivered = models.BooleanField(default=False)
    received = models.BooleanField(default=False)
    refund_requested = models.BooleanField(default=False)
    refund_granted = models.BooleanField(default=False)

    '''
    1. Item added to cart
    2. Adding a billing address
    (Failed checkout)
    3. Payment
    (Preprocessing, processing, packaging etc.)
    4. Being delivered
    5. Received
    6. Refunds
    '''

    def __str__(self):
        return self.user.username

    def get_total(self):
        total = 0
        for order_item in self.items.all():
            total += order_item.get_final_price()
        if self.coupon:
            total -= self.coupon.amount
        return total


class Address(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE)
    street_address = models.CharField(max_length=100)
    apartment_address = models.CharField(max_length=100)
    country = CountryField(multiple=False)
    zip = models.CharField(max_length=100)
    address_type = models.CharField(max_length=1, choices=ADDRESS_CHOICES)
    default = models.BooleanField(default=False)

    def __str__(self):
        return self.user.username

    class Meta:
        verbose_name_plural = 'Addresses'


class Payment(models.Model):
    stripe_charge_id = models.CharField(max_length=50)
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             on_delete=models.SET_NULL, blank=True, null=True)
    amount = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username


class Coupon(models.Model):
    code = models.CharField(max_length=15)
    amount = models.FloatField()

    def __str__(self):
        return self.code


class Refund(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    reason = models.TextField()
    accepted = models.BooleanField(default=False)
    email = models.EmailField()

    def __str__(self):
        return f"{self.pk}"


class FraudPolicy(models.Model):
    name = models.CharField(max_length=64, default='default-policy', unique=True)
    is_active = models.BooleanField(default=True)
    fraud_threshold = models.FloatField(default=0.5)
    block_transaction = models.BooleanField(default=True)
    ban_user = models.BooleanField(default=False)
    block_card_fingerprint = models.BooleanField(default=True)
    manual_review = models.BooleanField(default=True)
    send_email_alert = models.BooleanField(default=False)
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name='updated_fraud_policies'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} (threshold={self.fraud_threshold})"


class FraudBlocklist(models.Model):
    BLOCK_TYPE_CHOICES = (
        ('user_id', 'User ID'),
        ('card_fingerprint', 'Card Fingerprint'),
    )

    block_type = models.CharField(max_length=32, choices=BLOCK_TYPE_CHOICES)
    block_value = models.CharField(max_length=255)
    reason = models.CharField(max_length=255, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('block_type', 'block_value')

    def __str__(self):
        return f"{self.block_type}:{self.block_value}"


class FraudTransaction(models.Model):
    DECISION_CHOICES = (
        ('allow', 'Allow'),
        ('review', 'Manual Review'),
        ('block', 'Block'),
    )

    external_transaction_id = models.CharField(max_length=128, blank=True, null=True)
    external_user_id = models.CharField(max_length=128, blank=True, null=True)
    card_fingerprint = models.CharField(max_length=255, blank=True, null=True)
    amount = models.FloatField(default=0)
    currency = models.CharField(max_length=12, default='usd')
    ip_address = models.CharField(max_length=64, blank=True, null=True)
    device_info = models.CharField(max_length=255, blank=True, null=True)
    event_time = models.DateTimeField(blank=True, null=True)

    model_name = models.CharField(max_length=128, blank=True, null=True)
    model_version = models.CharField(max_length=64, blank=True, null=True)
    fraud_score = models.FloatField(default=0)
    is_fraud_prediction = models.BooleanField(default=False)
    decision = models.CharField(max_length=16, choices=DECISION_CHOICES, default='allow')
    decision_reasons = models.TextField(blank=True, null=True)
    policy_snapshot = models.TextField(blank=True, null=True)
    raw_payload = models.TextField(blank=True, null=True)
    processing_latency_ms = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"tx={self.external_transaction_id or self.pk} decision={self.decision} score={self.fraud_score:.4f}"


class FraudAlert(models.Model):
    SEVERITY_CHOICES = (
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    )
    STATUS_CHOICES = (
        ('open', 'Open'),
        ('closed', 'Closed'),
    )

    transaction = models.ForeignKey(FraudTransaction, on_delete=models.CASCADE, related_name='alerts')
    severity = models.CharField(max_length=16, choices=SEVERITY_CHOICES, default='medium')
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default='open')
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"alert={self.pk} severity={self.severity} status={self.status}"


class FraudModelReport(models.Model):
    model_name = models.CharField(max_length=128)
    model_version = models.CharField(max_length=64)
    is_champion = models.BooleanField(default=False)
    report_json = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_name}::{self.model_version} champion={self.is_champion}"


class FraudPolicyAuditLog(models.Model):
    policy = models.ForeignKey(FraudPolicy, on_delete=models.CASCADE, related_name='audit_logs')
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name='fraud_policy_audit_logs'
    )
    change_source = models.CharField(max_length=64, default='api')
    payload = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"policy={self.policy_id} source={self.change_source} at={self.created_at}"


def userprofile_receiver(sender, instance, created, *args, **kwargs):
    if created:
        userprofile = UserProfile.objects.create(user=instance)


post_save.connect(userprofile_receiver, sender=settings.AUTH_USER_MODEL)
