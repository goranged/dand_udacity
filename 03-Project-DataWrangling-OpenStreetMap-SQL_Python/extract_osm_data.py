
# coding: utf-8

# # OpenStreetMap Data Wrangling Project
# 
# ### Data extraction script
# 
# * ** Program: ** Data Analyst Nanodegree 
# * ** Student: ** Guillermo Naranjo
# * ** Date:** August 14, 2017
# 
# Extract OSM XML data for selected geolocation, apply a basic data quality process to validate and transform nodes, ways and attribute tags to predefined entities in CSV files. 
# 
# Based on OSM Case Study, the following script reads the OSM data file specified in OSM_PATH constant and extract the tags related to Nodes, Ways and its child tags. It also apply cleaning to extracted data, trasnformation into CSV and load to SQlite database.

# In[1]:


# ================================================== #
#     Importing required libraries                   #
# ================================================== #
import xml.etree.cElementTree as ET
from collections import defaultdict
import codecs
import cerberus
from unicode_dict_writer import unicode_dict_writer as udw
import schema
import re
import pprint
import sqlite3
import pandas as pd
import csv


# In[2]:


# ================================================== #
#     Global constants are defined                   #
# ================================================== #
OSM_PATH = "costa_rica_greater_metropolitan_area.osm"
DB_PATH = 'osm_costa_rica.db'

DROP_QUERY = """DROP TABLE IF EXISTS """
            
CREATE_QUERY_N ="""CREATE TABLE nodes (
    id INTEGER PRIMARY KEY NOT NULL,
    lat REAL,
    lon REAL,
    user TEXT,
    uid INTEGER,
    version INTEGER,
    changeset INTEGER,
    timestamp TEXT
);"""

CREATE_QUERY_NT = """CREATE TABLE nodes_tags (
    id INTEGER,
    key TEXT,
    value TEXT,
    type TEXT,
    FOREIGN KEY (id) REFERENCES nodes(id)
);
"""
CREATE_QUERY_W = """CREATE TABLE ways (
    id INTEGER PRIMARY KEY NOT NULL,
    user TEXT,
    uid INTEGER,
    version TEXT,
    changeset INTEGER,
    timestamp TEXT
);"""

CREATE_QUERY_WT = """CREATE TABLE ways_tags (
    id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    type TEXT,
    FOREIGN KEY (id) REFERENCES ways(id)
);"""

CREATE_QUERY_WN = """CREATE TABLE ways_nodes (
    id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    FOREIGN KEY (id) REFERENCES ways(id),
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);"""

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

SCHEMA = schema.schema

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\.\,\t\r\n]')
VALID_PHONE = re.compile(r'\(\d{3}\)\s*\d{8}') #using https://pythex.org/

TEXT_KEYS = {"name"}

MAPPING_CHARS = {
    ",":" de",
    ";":" o ",
    "en:":"Wikipedia ",
    "http://":"",
    "https://":"",
    "#":"número ",
    "&":"y",
    "$":"USD",
    ">":"",
    "<":"",
    "'":"",
    '"':"",
    "*":"",
    "+":"más",
    "%E9":"é",
    "%20":" ",
    "Desamaprados":"Desamparados",
    "Abast.":"Abastecedor",
    "Also knows as":"Conocido como",
    "Av.":"Avenida",
    "4ta": "cuarta",
    "Jr.":"Junior",
    "No.":"número ",
    "Dr.":"Doctor",
    "Dr":"Doctor",    
    "Dra.":"Doctora",
    "Dra":"Doctora",
    "S.A.":"SA",
    "Hnas.":"Hermanas",
    "Hnos.":"Hermanos",
    "MSc.":"Máster",    
    "Intl.":"Internacional",
    "!":"",
    "R.L.":"RL",
    "Urb.":"Urbanización",
    "Mo-Sa":"Lu-Sa",
    "Mo-Fr":"Lu-Vi",
    "Mo-Su":"Lu-Do",
    " Su ":" Do ",
    "Mo-Th":"Lu-Ju",
    "S.A:":"SA"
}

MAPPING = {
    "AM-PM":"AMPM",
    "Aeroscasillas":"Aerocasillas",
    "Azafran":"Azafrán",
    "Cerrajeria":"Cerrajería",
    "Coopeserviodores":"Coopeservidores",
    "JetBox":"Jetbox",
    "Macrobiotica":"Macrobiótica",
    "MegaSuper":"Megasuper",
    "Metropolis":"Metrópolis",
    "Metropoli":"Metrópolis",
    "Musammani":"Musmanni",
    "Musmani":"Musmanni",
    "Musmanny":"Musmanni",
    "Mussmani":"Musmanni",
    "Muswanni":"Musmanni",
    "Panaderia":"Panadería",
    "Pizzeria":"Pizzería",
    "Pulperia":"Pulpería",
    "Quiros":"Quirós",
    "Veterniaria":"Veterinaria",
    "Cartago1":"Cartago",
    "Rohmoser":"Rohrmoser"
}


# In[3]:


# ================================================== #
#               Validity Functions                   #
# ================================================== #

def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema """
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))

def is_problematic(kvalue):
    """ Validates if key value contains problematic characters based on PROBLEMCHARS regex """
    return PROBLEMCHARS.search(kvalue)  

def insert_text_separator(text):
    """ Transforms text values to UTF-8 """
    return text.encode('utf-8')

def right_type(key,value):
    """ Base on the item list, set the correct value datatype """
    if key in ['id','uid','changeset','version']:
        return int(value)
    elif key in ['lat','lon']:
        return float(value)
    else:
        return value
    
def log(id,key,value,newvalue):
    
    ''' Show estandar log message for changes applied '''
    
    print "TAG",id," [",key.upper(),",", value, "] fixed to [",key.upper(),",", newvalue.decode('utf-8'),"]"   


# In[4]:


# ================================================== #
#               Parser Functions                     #
# ================================================== #

def get_element(osm_file, tags=('node', 'way', 'relation')):
    
    """ Parse xml tree and filter to only include node, ways and relation nodes """
    
    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()

def get_key_type(k,default_tag_type='regular'):    
    
    """ Parse attribute k names to set the correct name and type if : exists in the name """
    
    klist = k.split(':')
    newkey = newtype = ""
    if len(klist) > 2:
        newkey = klist[1] + ":" + klist[2]
        newtype = klist[0]
    elif len(klist) == 2:
        newkey = klist[1]
        newtype = klist[0]
    else:
        newkey = klist[0]
        newtype = default_tag_type
    return newkey,newtype

def get_tags(_id, elements):
    
    """ get list of child tags dictionaries from node or ways tags """
    
    tags = []
    for tag in elements.iter("tag"):        
        if 'k' in tag.attrib:
            _key,_type = get_key_type(tag.attrib['k'])
            _value = tag.attrib['v'] 
            dtag = {}
            dtag['id'] = int(_id) 
            dtag['key'] = insert_text_separator(_key)
            dtag['value'] = insert_text_separator(_value)
            dtag['type'] = insert_text_separator(_type)
            tags.append(dtag)                   
    return tags

def get_nodes(id, elements):
    
    """ get list of child nodes dictionaries from ways tags """
    
    nodes = []
    i = 0
    for node in elements.iter("nd"):                
        dnode = {}
        dnode['id'] = int(id) 
        dnode['node_id'] = int(node.attrib['ref'])
        dnode['position'] = int(i) 
        nodes.append(dnode)        
        i+=1
    return nodes  


# In[5]:


# ================================================== #
#           Postal code cleaning functions         #
# ================================================== #

def fix_postcode(dic):
    
    """ returns the correct key value and type """
    
    dic['key'] = 'postcode'
    dic['type'] = 'addr'
    log(dic['id'],'postal_code','postal_code','postcode')


# In[6]:


# ================================================== #
#           Phone Numbers Cleaning Functions         #
# ================================================== #

def is_valid_phone(phone):
    
    """ check if phone comply with REGEX specification """
    
    return  VALID_PHONE.search(phone) is not None

def fix_phone(dic):
    
    """ update phone numbers in order to use expected format """
    
    phone = origphone = dic['value']
    # taken from: https://stackoverflow.com/questions/5658369/how-to-input-a-regex-in-string-replace 
    phone = re.sub(r"-|\+|,|\ |\(|\)|\ *","", phone).replace("tel:","")
    if len(phone) == 11:
        phone = "(506)" + phone[-8:]
    elif len(phone) == 8:
        phone = "(506)" + phone    
    dic['value'] = phone    
    log(dic['id'],dic['key'],origphone,dic['value'])


# In[7]:


def fix_chars(_value, chars=MAPPING_CHARS):   
    
    ''' Replacing multiple character with value ones'''
    
    #Taken from https://stackoverflow.com/questions/6116978/python-replace-multiple-strings
    
    chars = dict((re.escape(k), v) for k, v in chars.iteritems())
    pattern = re.compile("|".join(chars.keys()))
    text = pattern.sub(lambda m: chars[re.escape(m.group(0))], _value)
    return text


# In[8]:


# ================================================== #
#               Tag Names Cleaning Functions         #
# ================================================== #

def fix_name(dic, mapping=MAPPING):
    
    """ check tag name values against expected set, updating those with the expected value as required """    
    _value = dic['value']
    _key = dic['key']
    _type = dic['type']
    my_map = pd.Series(mapping)    

    if  _key in ['payment','network']: #special cases were commas are valid separator
        dic['value'] = _value.replace(","," o ")
        log(dic['id'],_key,_value,dic['value'])        
    elif _value in my_map:  #this replaces invalid places names with correct ones
        dic['value'] = my_map[_value]
        log(dic['id'],'name',_value,dic['value'])   
    elif is_problematic(_value): #this replaces invalid characters in values with valid ones
        dic['value'] = fix_chars(_value)
        log(dic['id'],_key,_value,dic['value'])        


# In[9]:


# ================================================== #
#          Master Cleaning Functions         #
# ================================================== #

def clean_tags(tags, textkeys=TEXT_KEYS):
    
    """ clean tags by fixing phone numbers, postal codes and names """
    
    for dic in tags:
        if dic['key'] == 'postal_code':            
            fix_postcode(dic)
        elif (dic['key'] == 'phone' or dic['key'] == 'fax') and not is_valid_phone(dic['value']):
            fix_phone(dic)
        else:
            fix_name(dic)

def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS):

    """ creates main dictionary structure based on OSM xml extract and clean as required """
    
    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []

    if element.tag == 'node':        
        
        for item in node_attr_fields:
            node_attribs[item] = right_type(item,element.attrib[item])
            
        tags = get_tags(node_attribs['id'],element)
        
        #This script apply cleaning process on nodes tags
        clean_tags(tags)
        
        return {'node': node_attribs, 'node_tags': tags}
    
    elif element.tag == 'way':
        
        for item in way_attr_fields:
            way_attribs[item] = right_type(item,element.attrib[item])
            
        way_nodes = get_nodes(way_attribs['id'],element)                        
        tags = get_tags(way_attribs['id'],element)
        
        #This script apply cleaning process on way tags
        clean_tags(tags)

        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# In[10]:


# ================================================== #
#       Database Creation and load Functions         #
# ================================================== #

def fill_table(table,conn):
    
    '''I used tentacles666 code to simplify data import. 
       https://tentacles666.wordpress.com/2014/11/14/python-creating-a-sqlite3-database-from-csv-files/'''
    
    conn.text_factory = str  # allows utf-8 data to be stored   
    curs = conn.cursor() 
    curs.execute('PRAGMA encoding="UTF-8";')
    
    # traverse the directory and process each .csv file    
    csvfile = table + ".csv"
    with open(csvfile, "rb") as f:
        reader = csv.reader(f)
 
        header = True
        for row in reader:
            if header:
                # gather column names from the first row of the csv
                header = False
                insertsql = "INSERT INTO %s VALUES (%s)" % (table,
                            ", ".join([ "?" for column in row ]))
 
                rowlen = len(row)
            else:
                # skip lines that don't have the right number of columns
                if len(row) == rowlen:
                    curs.execute(insertsql, row)   
                    
def fill_tables(conn):
    
    """ import csv to tables """
    
    curs = conn.cursor()
    fill_table("nodes",conn)
    fill_table("nodes_tags",conn)
    fill_table("ways",conn)
    fill_table("ways_tags",conn)
    fill_table("ways_nodes",conn)
    conn.commit()  
    curs.close()
    conn.close()
                    
def drop_tables(conn):
    
    """ drop tables is exist """
    
    curs = conn.cursor()
    curs.execute(DROP_QUERY + "nodes")
    curs.execute(DROP_QUERY + "nodes_tags")
    curs.execute(DROP_QUERY + "ways")
    curs.execute(DROP_QUERY + "ways_tags")
    curs.execute(DROP_QUERY + "ways_nodes")    
    conn.commit()   

def create_tables(conn):
    
    """ create tables """
    
    curs = conn.cursor()
    curs.execute(CREATE_QUERY_N)
    curs.execute(CREATE_QUERY_NT)
    curs.execute(CREATE_QUERY_W)
    curs.execute(CREATE_QUERY_WT)
    curs.execute(CREATE_QUERY_WN)
    conn.commit()


# In[11]:


# ================================================== #
#         Master creation and load DB Functions      #
# ================================================== #
def process_db(db,drop=DROP_QUERY):
    
    """ create osm database and import previously created csv """
    
    conn = sqlite3.connect(db)
    drop_tables(conn)
    create_tables(conn)
    fill_tables(conn)
    conn.close()
    print " DB creation and csv data import successful."


# In[ ]:


# ================================================== #
#       Main Function to Generate CSV                #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'wb') as nodes_file,         codecs.open(NODE_TAGS_PATH, 'wb') as nodes_tags_file,         codecs.open(WAYS_PATH, 'wb') as ways_file,         codecs.open(WAY_NODES_PATH, 'wb') as way_nodes_file,         codecs.open(WAY_TAGS_PATH, 'wb') as way_tags_file:

        nodes_writer = udw(nodes_file, NODE_FIELDS)
        node_tags_writer = udw(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = udw(ways_file, WAY_FIELDS)
        way_nodes_writer = udw(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = udw(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        i = 0
        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                i+=1
                if validate is True:
                    validate_element(el, validator)
                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])
                if i % 2000 == 1: print "\n", i-1,"TAGS PROCESSED. \n"
    print " OSM XML extraction, cleanup and csv creation successful."


# In[ ]:


if __name__ == "__main__":
    process_map(OSM_PATH, validate=True)    
    process_db(DB_PATH)

