# -*- coding: utf-8 -*-
# Fitr. A package for fitting reinforcement learning models to behavioural data
# Copyright (C) 2017 Abraham Nunes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# CONTACT INFO:
#   Abraham Nunes
#    Email: nunes@dal.ca
#
# ============================================================================
"""
Module for the base Generative Model object
"""

class GenerativeModel(object):
    """
    Base class for generative models

    Attributes
    ----------
    paramnames : dict
        Dictionary with two entries: 'long' which are strings representing parameter names, and 'code', which are strings denoting the parameter names as encoded in the Stan code
    model : string
        String representing Stan model code
    """
    def __init__(self):
        self.paramnames = {'long': [], 'code': []}
        self.model = ''
