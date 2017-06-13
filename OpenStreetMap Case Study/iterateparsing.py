#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Using iterative parsing to process the map file and
find out not only what tags are there, but also how many, to get the
feeling on how much of which data you can expect to have in the map.
The count_tags function should return a dictionary with the tag name as the key and number of times this tag can be encountered in 
the map as value.
"""
import xml.etree.cElementTree as ET
import pprint

def count_tags(filename):
        # YOUR CODE HERE
    tags = {}
    for event, elem in ET.iterparse(filename):
        if tags.has_key(elem.tag) == True:
            tags[elem.tag] += 1
        else:
            tags[elem.tag] = 1
    return tags

tags = count_tags('sample.osm')


print 'hellpo'
print tags