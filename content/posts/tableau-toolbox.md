---
title: "Tableau Automation Toolbox"
date: 2024-05-06T00:12:50-07:00
tags:
- Automation
- Python
- Unit Tests
- Object-Oriented Programming
- Data Structures
- Tableau
- Tableau Administration
- REST APIs
- requests Package
- List Comprehension
- Business Intelligence
- Data Visualization
metaAlignment: center
thumbnailImagePosition: "right"
thumbnailImage: https://img.freepik.com/free-vector/dashboard-service-abstract-concept-illustration_335657-3710.jpg?
# <a href="https://www.freepik.com/free-vector/dashboard-service-abstract-concept-illustration_12290891.htm#fromView=search&page=1&position=32&uuid=461a537f-81cd-4066-bc51-3122fb5c99b4">Image by vectorjuice</a> on Freepik
---

# Tableau Automation Toolbox: Overview

The role of a data scientist is wide-ranging; typically a data scientist at a startup will have to understand the entirety of an ETL pipeline (from web application to user reports) and be in communication with every team at the company. Reporting isn't always the most technologically exciting part of a data scientist's job but it's vital to understanding data and empowering all employees to promote data-driven decisions. All of the data collection and algorithms in the world can only be enhanced by measuring efficacy, reporting results and communicating the story told by the data to the wider users (internal and external) to inform your company’s strategy.

Alongside this noble endeavor, a data scientist may find themselves in the position of administering user credentials and managing user permissions and access to reports, and Tableau is a commonly used ETL reporting tool. Part of being a Tableau administrator involves managing user accounts, permissions, and security settings. This includes creating and deleting user accounts, assigning roles and permissions, and ensuring appropriate access controls are in place. This can be accomplished using the in-built Tableau GUI, but programmatic administration is faster and safer; with the API an administrator is less likely to make a mistake by clicking the wrong button or checking the wrong box, and users and there permissions can be instantly re-built, version-controlled, updated in mass and audited via unit and integration tests to get an instant picture of who has access to what data. 

This article will discuss the skills and concepts necessary to programmatically automate Tableau user administration and set up scripts to regularly audit user permissions using python and the Tableau REST API. 

{{< html >}}
<br>
{{< /html >}}

- [Tableau Automation Toolbox: Overview](#tableau-automation-toolbox-overview)
- [Tableau Administration Basics](#tableau-administration-basics)
- [Rest API Basics](#rest-api-basics)
- [Python Concepts](#python-concepts)
    - [Requests Package](#requests-package)
    - [Data Structures](#data-structures)
    - [List Comprehension](#list=comprehension)
    - [Object Oriented Programming](#object-oriented-programming)
- [Applied Examples with tableau_rest](#applied-examples-with-tableau_rest)
- [Further Applications](#further-applications)

# Tableau Administration Basics

On the Tableau server, *users* can be added and deleted, either one at a time using the GUI or en masse via a CSV file. Also using the GUI, users can also be added or removed into *groups* with similar data access needs to streamline *permissions* management. For more information on adding/removing groups/users on the Tableau server please see the Tableau documentation [here](https://help.tableau.com/current/server/en-us/users.htm).  

As a Tableau administrator, you can also control access to content such as workbooks, projects, and data sources by assigning permissions (also called *capabilities*) to users or groups by defining rules to specify the level of access users have  (e.g., read, write, manage) For more information on permissions and capabilities please see the Tableau documentation [here](https://help.tableau.com/current/server/en-us/permissions.htm).

A *workbook* is a .twb file that may contain multiple *dashboards* (interactive data visualizations). Individual Tableau *worksheets* consist of single data visualization objects that can be presented together interactively to form a dashboard. Workbooks can be published to projects on the Tableau Server. A *project* is a directory or folder on the Tableau Server where permissions for multiple workbooks can be set. 

# REST API Basics

A REST API (Representational State Transfer Application Programming Interface) is a set of rules and conventions for creating, accessing, and manipulating resources over the internet. REST APIs use HTTP protocols for communication and are based on principles of simplicity, scalability, and flexibility. In REST, *resources* are entities you want to interact with, such as users, products, or data. Each resource is represented by a unique URL (*endpoint*).

REST APIs use standard HTTP *methods* to perform different *CRUD* (Create, Read, Update, Delete) actions/operations on resources:
- **GET**: Retrieve data or resources (*read*)
- **POST**: Create new resources (*create*)
- **PUT**: Update existing resources (*update*)
- **DELETE**: Remove or delete resources (*delete*)

REST APIs return HTTP status codes to indicate the outcome of requests:
- 2xx: Success (e.g., 200 OK, 201 Created).
- 4xx: Client error (e.g., 400 Bad Request, 404 Not Found).
- 5xx: Server error (e.g., 500 Internal Server Error).

Understanding these codes will help you debug your API requests as you build them. For more information on the Tableau REST API see the Tableau documentation [here](https://help.tableau.com/current/api/rest_api/en-us/REST/rest_api_ref.htm). 

# Python Concepts
In combination with the REST API, there are several Python concepts necessary to understand in order to write automated administration scripts: the python requests package to formulate API requests and retrieve data, list comprehension syntax and basic data structures to store and query data retrieved from the API requests, and object-orientied programming concepts to organize your functions and classes for formulating/parsing API requests in a python script.  

## Requests Package
The requests package is a popular Python library for making HTTP requests easily and efficiently. It provides a simple and intuitive API for sending HTTP requests and handling responses. 
The package supports adding *query* parameters to HTTP requests using a dictionary or URL-encoded string and request *header* customization to specify content type, authentication, user-agent, and other headers. Data can be sent in PUT/POST requests using form data or JSON format using the `data` or `json ` parameters. The requests package can also easily handle various *authentication* methods such as basic, digest, or OAuth.
Responses from the API request can be parsed using the various response attributes such as status code, headers, content, JSON data, for example:
- `response.status_code`: HTTP status code of the response.
- `response.headers`: Headers returned in the response.
- `response.text`: The content of the response as a string.
- `response.json()`: Parse JSON content from the response.

When using the requests package to create specific functions for automating Tableau administration in a python script, it’s good practice to specify time limits for how long to wait for a response (*timeouts*). You can also create a `Session` object to persist certain parameters across multiple requests, such as cookies and authentication (so you can use the same authentication for your entire script)  and handle different types of request exceptions for easier debugging.
Overall, `requests` is a versatile and user-friendly package for making HTTP requests in Python, suitable for a wide range of use cases, from simple GET requests to complex data transfers and API interactions. For examples, syntax and an overview of all the package functionality, see the documentation for the requests package [here](https://requests.readthedocs.io/en/latest/)

## Data Structures
Data storage and retrieval is the key to managing and auditing Tableau users, groups and permissions. For example, a version-controlled table of tableau users and their groups and projects can be used such that the Tableau server is updated every time it is modified; a dictionary of projects with users and capabilities can be queried via the Tableau API to quickly check which Tableau users have specific kinds of access to which dashboards; and, XML code returned from a GET request to the Tableau server can be parsed to get a list of all users on the server. 
In Python, you can work with various data structures such as JSON, XML, and dictionaries to represent and manipulate data in different formats. 

**JSON** is a lightweight data interchange format that is easy for humans to read and write and easy for machines to parse and generate. 

```python
import json
### Parsing JSON: Convert JSON strings into Python dictionaries
json_str = '{"name": "Alice", "age": 30}'
data = json.loads(json_str)
### Convert Python data structures (e.g., dictionaries) into JSON strings
data = {"name": "Alice", "age": 30}
json_str = json.dumps(data)
```
For full functionality of the python json package check out the documentation [here](https://docs.python.org/3/library/json.html)

**XML** is a markup language used to represent hierarchical data and is commonly used for data exchange between systems.

```python
import xml.etree.ElementTree as ET
### Parse XML data from a file or string
xml_str = '<person><name>Alice</name><age>30</age></person>'
root = ET.fromstring(xml_str)
```
ElementTree Methods like find() and findall() can be used to navigate through the XML tree structure. XML can be written to a file or string using ET.tostring() or ET.write(). Tableau workbooks are written in XML and can be modified programmatically using the ElementTree package. For full functionality of the python ElementTree package check out the documentation [here](https://docs.python.org/3/library/xml.etree.elementtree.html)

**Dictionaries**
Dictionaries are a built-in data structure in Python used to store data in key-value pairs. They are unordered, mutable, and allow for fast lookups by key.
```python
### use curly braces {} or the dict constructor to create a dictionary
data = {"name": "Alice", "age": 30}
### access values in a dictionary using keys
name = data["name"]  # returns "Alice"
### add or modify key-value pairs
data["age"] = 31
data["city"] = "New York"
###  iterate through dictionaries using loops to access keys, values, or items (key-value pairs)
for key, value in data.items():
    print(f"{key}: {value}")
```

These data structures provide flexible and efficient ways to work with data in Python. For a detailed overview of data structures in python see this python guide [here](https://realpython.com/python-data-structures/)

## List Comprehension
List comprehensions provide a concise and readable way to access nested data by looping through keys and values in a dictionary using a single line of code. For example, given the following dictionary, lists of specific keys and values can be pulled out using the follonwg list comprehension:

```python
people = {
    "person1": {
        "name": "Alice",
        "age": 30,
        "city": "New York"
    },
    "person2": {
        "name": "Bob",
        "age": 25,
        "city": "Los Angeles"
    },
    "person3": {
        "name": "Charlie",
        "age": 35,
        "city": "Chicago"
    }
}

# Extracting all names from the nested dictionary using list comprehension
names = [people[person]["name"] for person in people]

# Printing the list of names
print(names)  # Output: ['Alice', 'Bob', 'Charlie']

# Extracting all ages from the nested dictionary using list comprehension
ages = [people[person]["age"] for person in people]

# Printing the list of ages
print(ages)  # Output: [30, 25, 35]

```
For a detailed overview of list comprehension in python see this python guide [here](https://realpython.com/list-comprehension-python/)

## Object Oriented Programming
Object-oriented programming (OOP) is a programming paradigm that organizes code around objects rather than functions and logic. In Python, OOP provides a way to model real-world entities and relationships using classes and objects. In brief, here are the object-oriented concepts that can be used to create a custom script for interacting with the Tableau API. 
- A **class** is a blueprint or template for creating objects. It defines the attributes (data) and methods (functions) that the objects created from the class will have.
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        print(f"Hello, my name is {self.name} and I'm {self.age} years old.")
```
- An **object** is an instance of a class. It represents an individual entity with its own state and behavior, as defined by the class.
```python
# Creating an object of the Person class
person1 = Person("Alice", 30)

# Accessing attributes and methods
print(person1.name)  # Output: Alice
person1.greet()  # Output: Hello, my name is Alice and I'm 30 years old.
```
- **Inheritance** allows a new class (subclass) to inherit attributes and methods from an existing class (superclass). This promotes code reuse and can help in organizing complex systems
```python
class Employee(Person):
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id
    
    def work(self):
        print(f"{self.name} is working.")
```

Object-oriented programming in Python provides a way to structure and manage code efficiently. By using classes and objects, developers can model complex systems, promote code reuse, and achieve better maintainability and scalability. For the purpose of the Tableau API, custom classes can be created with API calls using the python requests package, and the response from those API calls can be parsed and organized into data stuctures like lists and dictionaries to allow for the creation of scripts to quickly and easily audit users, groups and permissions. For more further information on object-oriented programming capabilities in python please see this guide [here](https://realpython.com/python3-object-oriented-programming/)

# Applied Examples with tableau_rest
Putting the concepts reviewed in brief (REST APIS, requests package, list comprehension and OOP) above all together, below are some examples of how to perform Tableau administrative tasks with custom python scripts.

Generally, each function in python is organized into the followng steps: 

1. Create url variable (can be found in API docs)
2. Build XMl request elements and subelements using Element tree (if applicable). XML request specifications can be found in API documentation
3. Convert Elementtree to string
4. Send request using python requests library
5. Check server response for error 
6. Convert server response to string, parses response to return relevant values (if applicable)

 Many of these examples are taken directly from a python module I have written and currently use to automate and audit Tableau administrative tasks. For more detailed examples please see the readme [here](https://github.com/cfrench575/tabula?tab=readme-ov-file). 

### Logging in
Here is an example function used to sign into the Tableau server using the Tableau API. The function returns the token, site_id and user_id that will be used for all API calls during the session. 

```python
import requests
import re
import xml.etree.ElementTree as ET
import urllib3

### create function

def sign_in(server, username, password, VERSION, xmlns, site=""):
    url = server + "/api/{0}/auth/signin".format(VERSION)
    # Builds the request as xml_object
    xml_request = ET.Element('tsRequest')
    credentials_element = ET.SubElement(xml_request, 'credentials', name=username, password=password)
    ET.SubElement(credentials_element, 'site', contentUrl=site)
    xml_request = ET.tostring(xml_request)
    # Make the request to server
    server_response = requests.post(url, data=xml_request, verify=False) ###add timeout=15
    _check_status(server_response, 200, xmlns)
    # # ASCII encode server response to enable displaying to console
    server_response = _encode_for_display(server_response.text)
    ## without _check_status function:
    # server_response=server_response.text.encode('ascii', errors="backslashreplace").decode('utf-8')
    # Reads and parses the response
    parsed_response = ET.fromstring(server_response)
    # Gets the auth token and site ID
    token = parsed_response.find('t:credentials', namespaces = xmlns).get('token')
    site_id = parsed_response.find('.//t:site', namespaces = xmlns).get('id')
    user_id = parsed_response.find('.//t:user', namespaces = xmlns).get('id')
    return token, site_id, user_id

### define variables - enter your own credentials 

server= "https://tableau.example.com"
username= "tableauusername"
password= "tableaupassword"
##if there is only one site you can leave this blank, otherwise specify site
site = ''
## version of tableau REST API
VERSION= '3.11'
## namespaces for API calls to the tableau server
xmlns = {'t': 'http://tableau.com/api'}

# sign in and store token, site_id and user_id to use for other methods
token, site_id, my_user_id = sign_in(server, username, password, VERSION, xmlns, site)
```
### Example 1: Query existing Tableau users on the server
First, create a class for easily accessing the returned user data from the Tableau API get request. The attributes for formulation the API request will be defined within the class. I've also added a method to the QueryUsers to convert a user name to a user_id - additional methods can be added depending on the specific use cases.

```python
class QueryUsers():
    def __init__(self, VERSION, site_id, token, server, xmlns):
        self.VERSION = VERSION
        self.site_id = site_id
        self.server = server
        self.xmlns = xmlns
        # GET /api/api-version/sites/site-id/users
        url = server + "/api/{0}/sites/{1}/users?pageSize=200".format(VERSION, site_id)
        # xml_request = 'none'
        server_response = requests.get(url, headers={'x-tableau-auth': token}, verify=False)
        server_response = _encode_for_display(server_response.text)
        parsed_response = ET.fromstring(server_response)
        self.users = parsed_response.findall('.//t:user', namespaces=xmlns)
        ### returns list of user names from GET request using list comprehension
        self.user_names= [user.get('name') for user in self.users]
        self.user_ids= [user.get('id') for user in self.users]

    def user_id_from_name(self, user_name):
        if user_name in self.user_names:
            user_id = [user.get('id') for user in self.users if user.get('name') == user_name]
        return user_id[0]

### create QueryUsers object
users_obj=tableau_rest.QueryUsers(VERSION, site_id, token, server, xmlns)

### user class method to get list of user names o the Tableau server
print(users_obj.user_names)
```      

### Example 2: Adding Tableau users from a datasource
After signing in, it is possible to add users to the Tableau server, either one at a time or in bulk from a datasource where you have managed your Tableau users and their credentials (either locally or on a server). First, we will create the methods for formulating the API requests and parsing the response. 

```python
import pandas as pd

### create a helper function for parsing xml

def _encode_for_display(text):
    """
    Encodes strings so they can display as ASCII in a Windows terminal window.
    This function also encodes strings for processing by xml.etree.ElementTree functions.
    Returns an ASCII-encoded version of the text.
    Unicode characters are converted to ASCII placeholders (for example, "?").
    """
    return text.encode('ascii', errors = "backslashreplace").decode('utf-8')

### create a function to add a user to the server

def add_user(VERSION, site_id, token, server, xmlns, user_name, site_role):
    # POST /api/api-version/sites/site-id/users
    url = server + "/api/{0}/sites/{1}/users".format(VERSION, site_id)
    xml_request = ET.Element('tsRequest')
    ### formulate the xml to ad your user to the server here
    user_element = ET.SubElement(xml_request, 'user', name=user_name, siteRole=site_role)
    xml_request=ET.tostring(xml_request)
    ## adding the user to the server will return some xml; print the parsed xml to verify the user was added
    server_response = requests.post(url, data=xml_request, headers={'x-tableau-auth': token}, verify=False)
    server_response = _encode_for_display(server_response.text)
    parsed_response = ET.fromstring(server_response)
    new_user = parsed_response.findall('.//t:user', namespaces=xmlns)
    for x in new_user:
        user_name= x.get('name')
        site_role= x.get('SiteRole')
        print(user_name, site_role)
    ### return the user_name and site_role for the user just added to the server 
    return user_name, site_role

### create function to update user on the server. Adding the user and updating the user email and password are performed in separate steps

def update_user(VERSION, site_id, token, server, xmlns, user_id, new_name, new_email, new_password, new_siterole):
    # PUT /api/api-version/sites/site-id/users/user-id
    url = server + "/api/{0}/sites/{1}/users/{2}".format(VERSION, site_id, user_id)
    xml_request = ET.Element('tsRequest')
    user_element = ET.SubElement(xml_request, 'user', fullName = new_name, email = new_email, password = new_password, siteRole = new_siterole)  
    xml_request=ET.tostring(xml_request)
    server_response = requests.put(url, data=xml_request, headers={'x-tableau-auth': token}, verify=False)

# Example dataframe (for your administrative task, import your own data)

data = {
    "username": ["user1", "user2", "user3"],
    "licenselevel": ["Viewer", "Explorer", "Creator"],
    "emailaddress": ["user1@example.com", "user2@example.com", "user3@example.com"],
    "password": ["password1", "password2", "password3"]
}

df = pd.DataFrame(data)
```
Now, you can loop through your dataframe and add users to the server programmatically from a datasource. 

```python
### iterate through dataframe and add users to the server
for i, row in df.iterrows():
        ### only add a user if the user is not yet on the Tableau server (i.e not in the user_names list we created from the QueryUsers class)
        if row.username not in users_obj.user_names:
            ## adds new user not yet on server
            add_user(VERSION, site_id, token, row.username, row.licenselevel)
            ## after adding the user to the server, now wait 45 seconds for cache to clear
            time.sleep(45)
            ## query users now including the new user just added
            new_users_obj = QueryUsers(SET.VERSION, site_id, token)
            user_id = new_users_obj.user_id_from_name(row.username)
            ## after adding user to the server, update user to set password, email address
            update_user(VERSION, site_id, token, user_id, row.username, row.emailaddress, row.password, row.licenselevel)
            print("add " + row.username + " to server")
```
### Example 2: Adding users to groups
Using the classes and functions already defined, we can add additional support for adding groups to the server, and adding users to groups. Since permissions can be shared within a group, adding users to groups does simplify permissions/capabilities management. First, create a class to return information about groups on your Tableau server using the same logic present in the QueryUsers class. Then, create a function to add a group to the Tableau server, and to add a user to a group.

```python
### create QueryGroups class
class QueryGroups():
    """
    performs single API call that returns xml data for groups. xml is parsed using associated methods
    """
    def __init__(self, VERSION, site_id, token, server, xmlns):
        self.VERSION = VERSION
        self.site_id = site_id
        self.server = server
        self.xmlns = xmlns
        # GET /api/api-version/sites/site-id/groups/
        url = server + "/api/{0}/sites/{1}/groups".format(VERSION, site_id)
        # xml_request = 'none'
        server_response = requests.get(url, headers={'x-tableau-auth': token}, verify=False)
        server_response = _encode_for_display(server_response.text)
        parsed_response = ET.fromstring(server_response)
        self.groups = parsed_response.findall('.//t:group', namespaces=xmlns)

        self.group_names= [group.get('name') for group in self.groups]
        self.group_ids= [group.get('id') for group in self.groups]

    def group_id_from_name(self, group_name):
        _check_user_input(group_name, self.group_names)
        if group_name in self.group_names:
            group_id = [group.get('id') for group in self.groups if group.get('name') == group_name]
        return group_id[0]

### function to add group to server

def add_group(VERSION, site_id, token, server, xmlns, group_name, min_site_role = 'Viewer'):
    # POST /api/api-version/sites/site-id/groups
    url = server + "/api/{0}/sites/{1}/groups".format(VERSION, site_id)
    xml_request = ET.Element('tsRequest')
    group_element = ET.SubElement(xml_request, 'group', name=group_name, minimumSiteRole=min_site_role)
    xml_request=ET.tostring(xml_request)
    server_response = requests.post(url, data=xml_request, headers={'x-tableau-auth': token}, verify=False)
    server_response = _encode_for_display(server_response.text)
    parsed_response = ET.fromstring(server_response)
    new_group = parsed_response.findall('.//t:group', namespaces=xmlns)
    for x in new_group:
        group_name= x.get('name')
        group_id= x.get('id')
        minsiterole= x.get('minimumSiteRole')
        print(group_name, group_id, minsiterole) 
    return group_name, group_id, minsiterole

### function to add user to group

def add_user_to_group(VERSION, site_id, token, server, xmlns, group_id, user_id):
    # /api/api-version/sites/site-id/groups/group-id/users
    url = server + "/api/{0}/sites/{1}/groups/{2}/users".format(VERSION, site_id, group_id)
    xml_request = ET.Element('tsRequest')
    user_element = ET.SubElement(xml_request, 'user', id=user_id)
    xml_request=ET.tostring(xml_request)
    server_response = requests.post(url, data=xml_request, headers={'x-tableau-auth': token}, verify=False)
    server_response = _encode_for_display(server_response.text)
    parsed_response = ET.fromstring(server_response)
    new_user = parsed_response.findall('.//t:user', namespaces=xmlns)
    for x in new_user:
        user_name= x.get('name')
        user_id= x.get('id')
        print(user_name, user_id) 
    return user_name, user_id

# Update example dataframe to include group membership for user (for your administrative task, import your own data)

data = {
    "username": ["user1", "user2", "user3"],
    "licenselevel": ["Viewer", "Explorer", "Creator"],
    "emailaddress": ["user1@group1.com", "user2@group2.com", "user3@egroup3.com"],
    "password": ["password1", "password2", "password3"]
    "group": ["group1", "group2", "group3"]
}

df = pd.DataFrame(data)
```
Taking the same loop from above to add users to the server, add in logic to also add the user to a group:

```python
### create groups object
groups_obj=QueryGroups(VERSION, site_id, token, server, xmlns)

or i, row in df.iterrows():
        if row.username not in users_obj.user_names:
            add_user(SET.VERSION, site_id, token, row.username, row.licenselevel)
            time.sleep(45)
            new_users_obj = tableau_rest.QueryUsers(SET.VERSION, site_id, token)
            user_id = new_users_obj.user_id_from_name(row.username)
            tableau_rest.update_user(SET.VERSION, site_id, token, user_id, row.username, row.emailaddress, row.password, row.licenselevel)
            print("add " + row.username + " to server")
            group_in_table = row.group
            ## if group is not on the server, add group to server:
                # add_group(VERSION, site_id, token, group_in_table)
                # print("add " + group_in_table + " to server")
            # Add user to group here
            server_group_id=groups_obj.group_id_from_name(group_in_table)
            add_user_to_group(VERSION, site_id, token, server_group_id, user_id)
            print("add " + row.username + " to " + group_in_table)
```
### Example 3: Unit tests for auditing 
After creating classes and functions in python for querying the Tableau API, now unit and integration tests can be created to check user permissions. For example, if permissions are set per group and only users with a certain email string are allowed per group, you can run unit tests:

```python
## define a function to return all users in a group on the Tableau server

def users_in_group(VERSION, site_id, token, group_id, server, xmlns):
    #GET /api/api-version/sites/site-id/groups/group-id/users
    url = server + "/api/{0}/sites/{1}/groups/{2}/users".format(VERSION, site_id, group_id)
    server_response = requests.get(url, headers={'x-tableau-auth': token}, verify=False)
    server_response = _encode_for_display(server_response.text)
    parsed_response = ET.fromstring(server_response)
    users = parsed_response.findall('.//t:user', namespaces=xmlns)
    group_users=[]
    for user in users:
        group_users.append(user.get('name'))
    return group_users

### create groups object
groups_obj=QueryGroups(VERSION, site_id, token, server, xmlns)

import unittest

### check email string for each user for each group on server
class TestUserGroups(unittest.TestCase):
    def test_users_group1(self):
        users=users_in_group(VERSION, site_id, token, groups_obj.group_id_from_name('group1'), server, xmlns)
        for user in users:
            self.assertTrue("@group1.com" in user, "group1 input string does not match; need to check again")

    def test_users_group2(self):
        users=users_in_group(VERSION, site_id, token, groups_obj.group_id_from_name('group2'), server, xmlns)
        for user in users:
            self.assertTrue("@group2.com" in user, "group2 input string does not match; need to check again")

    def test_users_group3(self):
        users=users_in_group(VERSION, site_id, token, groups_obj.group_id_from_name('group3'), server, xmlns)
        for user in users:
            self.assertTrue("@group3.com" in user, "group3 input string does not match; need to check again")

if __name__ == '__main__':
    unittest.main()
```
This is just one example; there are infinite creative ways to audit users, permissions and access with unit and integration tests. With unit tests, mistakes can be caught far more quickly than checking/verifying each individual user by hand in the Tableau server GUI. For more information on unit tests in python please see the documentation [here](https://docs.python.org/3/library/unittest.html)

### Organizing your code
To keep track of global variables, it can be helpful to organize them in a `settings.py` file. For example:

```python
import os
dirname = os.path.dirname(__file__)
VERSION= '3.11'
username = 'tableauusername'
password = 'tableaupassword'
server = "https://tableau.example.com"
site=''
xmlns = {'t': 'http://tableau.com/api'}

### these are inputs allowed by the Tableau rest API - they might need to be updated as the Tableau server gets updated
VALID_PERMISSIONS_OBJECTS = {"project", "workbook", "datasource", "flow", "metric"}
VALID_PERMISSIONS_MODES = {"Allow", "Deny"}
VALID_PROJECT_CAPABILITIES = {"Read", "Write"}
VALID_WORKBOOK_CAPABILITIES = {'AddComment', 'ChangeHierarchy', 'ChangePermissions', 'Delete', 'ExportData', 'ExportImage', 'ExportXml', 'Filter', 'Read', 'ShareView', 'ViewComments', 'ViewUnderlyingData', 'WebAuthoring', 'Write'}
VALID_DATASOURCE_CAPABILITIES = {'ChangePermissions', 'Connect', 'Delete', 'ExportXml', 'Read', 'Write'}

### projects on the tableau server tracked via unit/integration tests
projects_to_track= ['project1', 'project2', 'project3', 'project4', 'project5']

groups_to_track= ['group1', 'group2', 'group3']

## permissions are synced on the server using this file
PATH_TO_MAIN_PERMISSIONS = os.path.join(dirname, "permissions.json")

### single excel file used to keep track of all users
PATH_TO_USER_FILE = os.path.join(dirname, "usersandgroups.csv")
```
Variables can be referenced from python scrpits, or the command line. This will keep global variables consistent in addition to being easily referenced and modified in a single location. 

A permissions file could be a datatable, excel file, or even a json file that might look something like this: 
```python
     {
  "groups": {
      "groupuuid": {
          "datasource": {},
          "flow": {},
          "metric": {},
          "project": {
              "Read": "Allow"
          },
          "workbook": {
              "AddComment": "Allow",
              "ExportData": "Allow",
              "ExportImage": "Allow",
              "Filter": "Allow",
              "Read": "Allow",
              "ShareView": "Allow",
              "ViewComments": "Allow",
              "ViewUnderlyingData": "Allow",
              "WebAuthoring": "Allow"
          }
      },
  "users": {
      "useruuid": {
          "datasource": {},
          "flow": {},
          "metric": {},
          "project": {
              "Read": "Allow"
          },
          "workbook": {
              "AddComment": "Allow",
              "ExportData": "Allow",
              "ExportImage": "Allow",
              "Filter": "Allow",
              "Read": "Allow",
              "ShareView": "Allow",
              "ViewComments": "Allow",
              "ViewUnderlyingData": "Allow",
              "WebAuthoring": "Allow"
          }
      }
```
The permissions file can be modified manually or programmatically and then changes can be synced with the Tableau server. Managing permissions this way minimizes mistakes that can be made with repetative entries in the Tableau permissions GUI. It also makes it easier to bulk add/update/delete permissions and the check consistency of permissions across the server.

# Further Applications
Using the Tableau server and python, it is possible to query, add, update and delete users, groups, projects, workbooks and permissions - for a complete example of this functionality, see the tableau_rest module located on my github [here](https://github.com/cfrench575/tabula/tree/main). 

This functionality can serve as the backbone for custom, fully automated Tableau administration; unit and integration tests can be built out to quickly audit users and permissions. Permissions and user data structures (locally or on a server) can be used for user management instead of the Tableau server GUI, which can be modified via the command line or scripts and automatically synced to the Tableau server. Tableau workbook XML can also be modified via the Tableau API; this is not addressed in the current article, but can be used for managing Tableau workbooks in bulk for templating reports, auditing calculated fields, updating calculated fields and swapping out datasources. 

The use-cases for reporting in an organization can be very specific and idiosyncratic. Building out your own custom reporting automation with the above concepts will hopefully save time and minimize manual mistakes. Happy hacking!