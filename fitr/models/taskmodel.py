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
""" Module containing the base class for task models """

import io
import os
from .generative_model import GenerativeModel

class TaskModel(object):
    """
    Base class for models of various tasks

    Attributes
    ----------
    gm : GenerativeModel
        Contains stan code and parameter identifiers

    Methods
    -------
    set_gm(self, path, paramnames_long, paramnames_code)

    """
    def __init__(self):
        self.gm = None

    def set_gm(self, path, paramnames_long, paramnames_code):
        """
        Instantiates the generative model for the task

        Parameters
        ----------
        filepath : str
            Path to the Stan file. Will be appended to 'stancode/'.
        paramnames_long : list
            List of strings with names for the parameters (for plots, etc.)
        paramnames_code : list
            List of strings representing the parameter names in the stan file

        Returns
        -------
        GenerativeModel

        """
        self.gm = GenerativeModel()

        # Get the model code
        this_dir, this_filename = os.path.split(__file__)
        DATA_PATH = os.path.join(this_dir, path)
        try:
            with io.open(DATA_PATH, 'rt') as f:
                model_code = f.read()
        except:
            raise IOError("Unable to read file specified by `file`.")

        self.gm.model = model_code
        self.gm.paramnames['long'] = paramnames_long
        self.gm.paramnames['code'] = paramnames_code
