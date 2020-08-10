# -*- coding: utf-8 -*-

class SomeClass():

    def __init__(self, some_value):
        self.some_value = some_value

    def multiply(self, multiplier):
        calc_result = self.some_value * multiplier
        return calc_result