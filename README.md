
# Technical task
  
*The objective* is, to predict whether a customer is going to return (order again from us) in *the upcoming 6 months* or not.  
So the main idea is, *to create an algorithm to correctly predict the **_is_returning_customer_*** label (see details [here](#Labeled-data)) based on the [provided data](#Data-dictionary).  
  
You can solve this exercise using the Python data ecosystem with the usual well-known tools and libraries.  
  
## Data dictionary  
  
There are [two data samples](./data/) provided as (gunzipped) CSV files.  
  
### Order data  
  
The [order dataset](./data/machine_learning_challenge_order_data.csv.gz) contains the history of orders placed by customers acquired between 2015-03-01 and 2017-02-28.
  
The dataset columns definition:  
  
|Column|Description|  
|---|---|  
|*customer_id*|Unique customer ID.|  
|*order_date*|Local date of the order.|  
|*order_hour*|Local hour of the order.|  
|*customer_order_rank*|Number of a successful order counted in chronological order starting with 1 (an empty value would correspond to a failed order).|  
|*is_failed*|**0** if the order succeeded.<br>**1** if the order failed.|  
|*voucher_amount*|The discounted amount if a voucher (discount) was used at order's checkout.|  
|*delivery_fee*|Fee charged for the delivery of the order (if applicable).|  
|*amount_paid*|Total amount paid by the customer (the *voucher_amount* is already deducted and the *delivery_fee* is already added).|  
|*restaurant_id*|Unique restaurant ID.|  
|*city_id*|Unique city ID.|  
|*payment_id*|Identifies the payment method the customer has chosen (such as cash, credit card, PayPal, ...).|  
|*platform_id*|Identifies the platform the customer used to place the order (web, mobile app, mobile web, …).|  
|*transmission_id*|Identifies the method used to place the order to the restaurant (fax, email, phone, and different kinds of proprietary devices or point-of-sale systems).|  
  
### Labeled data  
  
The [labeled dataset](./data/machine_learning_challenge_labeled_data.csv.gz) flags whether the customers placed at least one order within 6 months after 2017-02-28 or not.  

The dataset columns definition:  
|Column|Description|  
|---|---|  
|*customer_id*|Unique customer ID.|  
|*is_returning_customer*|**0** if the customer did not return (did not order again) in the 6 months after 2017-02-28.<br>**1** if the customer returned (ordered again) at least once after 2017-02-28.|  
