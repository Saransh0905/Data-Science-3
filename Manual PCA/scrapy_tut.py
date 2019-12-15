#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 23:31:17 2019

@author: saransh
"""

import scrapy 
class JokesSpider(scrapy.Spider):
    name = 'jokes'
    start_url = ['http://www.laughfactory.com/jokes/family-jokes']
    def parse(self,response):
        
            
            
        for joke in response.xpath("//div[@class='jokes']"):
            with open ('new_data','wb') as nd:
                nd.write(joke.xpath(".//div[@class = 'joke-text']/p"))
            yield {
                    'joke_text':joke.xpath(".//div[@class = 'joke-text']/p").extract_first()
                    }
        next_page = response.xpath("//link[@class = 'next']/a/@href").extract_first()
        if next_page is not None:
            next_page_link = response.urljoin(next_page)
            yield 
            {scrapy.Request(url = next_page_link,callback = self.parse)}
            
