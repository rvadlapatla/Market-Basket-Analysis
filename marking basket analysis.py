#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "plotly_white"

data = pd.read_csv(r"C:\Users\User\Downloads\market_basket_dataset.csv")
print(data.head())


# In[2]:


print(data.isnull().sum())


# In[3]:


data.describe()


# In[4]:


fig =px.histogram(data,x="Itemname",title="total of Items")
fig.show()


# In[5]:


item_popularity=data.groupby('Itemname')['Quantity'].sum().sort_values(ascending =False)
top_n=15
fig =go.Figure()
fig.add_trace(go.Bar(x=item_popularity.index[:top_n],y=item_popularity.values[:top_n],
                    text =item_popularity.values[:top_n],textposition ='auto',marker=dict(color='blue')))
fig.update_layout(title=f'Top{top_n}Most popular Items ')
fig.show()


# In[6]:


customer_behavior =data.groupby('CustomerID').agg({'Quantity':'mean','Price':'sum'}).reset_index()
table_data=pd.DataFrame({'CustomerID':customer_behavior["CustomerID"],
                        'Average Quantity':customer_behavior['Quantity'],
                       'Total_Spending':customer_behavior['Price']})
fig=go.Figure()
fig.add_trace(go.Scatter(x=customer_behavior["CustomerID"],y=customer_behavior['Quantity'],
                        mode='markers',text=customer_behavior['CustomerID'],marker =dict(size=10,color="coral")))
#add table
fig.add_trace(go.Table(header=dict(values=['CustomerID','Average Quantity','Total_Spending']),
             cells=dict(values=[table_data['CustomerID'],table_data['Average Quantity'],table_data['Total_Spending']]),))
fig.update_layout(title="custmer behavior",xaxis_title="Average Quantity",yaxis_title='Total_Spending')
fig.show()
                                                             


# Now, let’s use the Apriori algorithm to create association rules. The Apriori algorithm is used to discover frequent item sets in large transactional datasets. It aims to identify items that are frequently purchased together in transactional data. It helps uncover patterns in customer behaviour, allowing businesses to make informed decisions about product placement, promotions, and marketing. Here’s how to implement Apriori to generate association rules:

# In[7]:


from  mlxtend.frequent_patterns import apriori,association_rules

#grop items by billno
basket =data.groupby("BillNo")["Itemname"].apply(list).reset_index()
#print(basket)

#encode item as binary variables using one hot encoding
basket_encode=basket["Itemname"].str.join("|").str.get_dummies("|")
##print(basket_encode)

frequent_item=apriori(basket_encode,min_support=0.01,use_colnames=True)
#print(frequent_item)
rules =association_rules(frequent_item,metric ='lift',min_threshold=0.5)
print(rules[['antecedents','consequents','support',"lift"]].head(20))


# In[ ]:




