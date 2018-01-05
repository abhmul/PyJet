# -*- coding: utf-8 -*-
'''
General documentation architecture (Still more work to do here):
Home
Index
- Getting started
    Getting started
    FAQ
- Models
    About PyJet models
        explain designing and building a model
        explain weight saving, weight loading
    SLModel
- Layers
    About PyJet layers
        explain common layer functions: get_weights, set_weights, get_config
        explain usage on pytorch tensors
    Core Layers
    Convolutional Layers
    Pooling Layers
    Recurrent Layers
    Layer Wrappers
    Writing your own PyJet layers
- Preprocessing
    Image Preprocessing
Data
Losses
Metrics
Callbacks
Backend
Scikit-learn API
Utils
Contributing
'''
from __future__ import print_function
from __future__ import unicode_literals

import re
import inspect
import os
import shutil

from pyjet import utils
from pyjet import data
from pyjet import layers
from pyjet import callbacks
from pyjet import models
from pyjet import losses
from pyjet import metrics
from pyjet import backend
from pyjet import preprocessing

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding('utf8')


EXCLUDE = {}

PAGES = [
    {
        'page': 'models/SLModel.md',
        'functions': [
            models.SLModel.cast_input_to_torch,
            models.SLModel.cast_target_to_torch,
            models.SLModel.cast_output_to_numpy,
            models.SLModel.forward,
            models.SLModel.train_on_batch,
            models.SLModel.validate_on_batch,
            models.SLModel.predict_on_batch,
            models.SLModel.fit_generator,
            models.SLModel.validate_generator,
            models.SLModel.predict_generator,
            models.SLModel.load_state,
            models.SLModel.save_state,
        ],
    },
    {
        'page': 'layers/core.md',
        'classes': [
            layers.FullyConnected,
            layers.Flatten,
        ],
    },
    {
        'page': 'layers/convolutional.md',
        'classes': [
            layers.Conv1D,
            layers.Conv2D,
            layers.Conv3D,
        ],
    },
    # {
    #     'page': 'layers/pooling.md',
    #     'classes': [
    #     ],
    # },
    {
        'page': 'layers/recurrent.md',
        'classes': [
            layers.SimpleRNN,
            layers.GRU,
            layers.LSTM,
        ],
    },
    {
        'page': 'layers/functions.md',
        'all_module_functions': [layers.functions]
    },
    {
        'page': 'metrics.md',
        'all_module_functions': [metrics],
    },
    {
        'page': 'losses.md',
        'all_module_functions': [losses],
    },
    {
        'page': 'callbacks.md',
        'all_module_classes': [callbacks],
    },
    {
        'page': 'backend.md',
        'all_module_functions': [backend],
    },
    {
        'page': 'preprocessing/image.md',
        'classes': [
            preprocessing.image.ImageDataGenerator,
        ],
        'functions': [
            preprocessing.image.ImageDataGenerator.standardize,
            preprocessing.image.ImageDataGenerator.random_transform,
        ]
    },
    {
        'page': 'data.md',
        'classes': [
            data.Dataset,
            data.NpDataset,
            data.HDF5Dataset,
            data.TorchDataset,
            data.DatasetGenerator,
        ]
    }
    # {
    #     'page': 'utils.md',
    # },
]

ROOT = 'http://pyjet.io/'


def get_earliest_class_that_defined_member(member, cls):
    ancestors = get_classes_ancestors([cls])
    result = None
    for ancestor in ancestors:
        if member in dir(ancestor):
            result = ancestor
    if not result:
        return cls
    return result


def get_classes_ancestors(classes):
    ancestors = []
    for cls in classes:
        ancestors += cls.__bases__
    filtered_ancestors = []
    for ancestor in ancestors:
        if ancestor.__name__ in ['object']:
            continue
        filtered_ancestors.append(ancestor)
    if filtered_ancestors:
        return filtered_ancestors + get_classes_ancestors(filtered_ancestors)
    else:
        return filtered_ancestors


def get_function_signature(function, method=True):
    wrapped = getattr(function, '_original_function', None)
    if wrapped is None:
        signature = inspect.getargspec(function)
    else:
        signature = inspect.getargspec(wrapped)
    defaults = signature.defaults
    if method:
        args = signature.args[1:]
    else:
        args = signature.args
    if defaults:
        kwargs = zip(args[-len(defaults):], defaults)
        args = args[:-len(defaults)]
    else:
        kwargs = []
    st = '%s.%s(' % (function.__module__, function.__name__)

    for a in args:
        st += str(a) + ', '
    for a, v in kwargs:
        if isinstance(v, str):
            v = '\'' + v + '\''
        st += str(a) + '=' + str(v) + ', '
    if kwargs or args:
        signature = st[:-2] + ')'
    else:
        signature = st + ')'

    if not method:
        # Prepend the module name.
        signature = function.__module__ + '.' + signature
    return post_process_signature(signature)


def get_class_signature(cls):
    try:
        class_signature = get_function_signature(cls.__init__)
        class_signature = class_signature.replace('__init__', cls.__name__)
    except (TypeError, AttributeError):
        # in case the class inherits from object and does not
        # define __init__
        class_signature = cls.__module__ + '.' + cls.__name__ + '()'
    return post_process_signature(class_signature)


def post_process_signature(signature):
    parts = re.split('\.(?!\d)', signature)
    if len(parts) >= 4:
        if parts[1] == 'layers':
            signature = 'pyjet.layers.' + '.'.join(parts[3:])
        if parts[1] == 'utils':
            signature = 'pyjet.utils.' + '.'.join(parts[3:])
        if parts[1] == 'backend':
            signature = 'pyjet.backend.' + '.'.join(parts[3:])
    return signature


def class_to_docs_link(cls):
    module_name = cls.__module__
    assert module_name[:6] == 'pyjet.'
    module_name = module_name[6:]
    link = ROOT + module_name.replace('.', '/') + '#' + cls.__name__.lower()
    return link


def class_to_source_link(cls):
    module_name = cls.__module__
    assert module_name[:6] == 'pyjet.'
    path = module_name.replace('.', '/')
    path += '.py'
    line = inspect.getsourcelines(cls)[-1]
    link = ('https://github.com/PyJet/'
            'tree/master/pyjet/' + path + '#L' + str(line))
    return '[[source]](' + link + ')'


def code_snippet(snippet):
    result = '```python\n'
    result += snippet + '\n'
    result += '```\n'
    return result


def count_leading_spaces(s):
    ws = re.search('\S', s)
    if ws:
        return ws.start()
    else:
        return 0


def process_docstring(docstring):
    # First, extract code blocks and process them.
    code_blocks = []
    if '```' in docstring:
        tmp = docstring[:]
        while '```' in tmp:
            tmp = tmp[tmp.find('```'):]
            index = tmp[3:].find('```') + 6
            snippet = tmp[:index]
            # Place marker in docstring for later reinjection.
            docstring = docstring.replace(
                snippet, '$CODE_BLOCK_%d' % len(code_blocks))
            snippet_lines = snippet.split('\n')
            # Remove leading spaces.
            num_leading_spaces = snippet_lines[-1].find('`')
            snippet_lines = ([snippet_lines[0]] +
                             [line[num_leading_spaces:]
                             for line in snippet_lines[1:]])
            # Most code snippets have 3 or 4 more leading spaces
            # on inner lines, but not all. Remove them.
            inner_lines = snippet_lines[1:-1]
            leading_spaces = None
            for line in inner_lines:
                if not line or line[0] == '\n':
                    continue
                spaces = count_leading_spaces(line)
                if leading_spaces is None:
                    leading_spaces = spaces
                if spaces < leading_spaces:
                    leading_spaces = spaces
            if leading_spaces:
                snippet_lines = ([snippet_lines[0]] +
                                 [line[leading_spaces:]
                                  for line in snippet_lines[1:-1]] +
                                 [snippet_lines[-1]])
            snippet = '\n'.join(snippet_lines)
            code_blocks.append(snippet)
            tmp = tmp[index:]

    # Format docstring section titles.
    docstring = re.sub(r'\n(\s+)# (.*)\n',
                       r'\n\1__\2__\n\n',
                       docstring)
    # Format docstring lists.
    docstring = re.sub(r'    ([^\s\\\(]+):(.*)\n',
                       r'    - __\1__:\2\n',
                       docstring)

    # Strip all leading spaces.
    lines = docstring.split('\n')
    docstring = '\n'.join([line.lstrip(' ') for line in lines])

    # Reinject code blocks.
    for i, code_block in enumerate(code_blocks):
        docstring = docstring.replace(
            '$CODE_BLOCK_%d' % i, code_block)
    return docstring

print('Cleaning up existing sources directory.')
if os.path.exists('sources'):
    shutil.rmtree('sources')

print('Populating sources directory with templates.')
for subdir, dirs, fnames in os.walk('templates'):
    for fname in fnames:
        new_subdir = subdir.replace('templates', 'sources')
        if not os.path.exists(new_subdir):
            os.makedirs(new_subdir)
        if fname[-3:] == '.md':
            fpath = os.path.join(subdir, fname)
            new_fpath = fpath.replace('templates', 'sources')
            shutil.copy(fpath, new_fpath)

# Take care of index page.
readme = open('../README.md').read()
index = open('templates/index.md').read()
index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
f = open('sources/index.md', 'w')
f.write(index)
f.close()

print('Starting autogeneration.')
for page_data in PAGES:
    blocks = []
    classes = page_data.get('classes', [])
    for module in page_data.get('all_module_classes', []):
        module_classes = []
        for name in dir(module):
            if name[0] == '_' or name in EXCLUDE:
                continue
            module_member = getattr(module, name)
            if inspect.isclass(module_member):
                cls = module_member
                if cls.__module__ == module.__name__:
                    if cls not in module_classes:
                        module_classes.append(cls)
        module_classes.sort(key=lambda x: id(x))
        classes += module_classes

    for cls in classes:
        subblocks = []
        signature = get_class_signature(cls)
        subblocks.append('<span style="float:right;">' +
                         class_to_source_link(cls) + '</span>')
        subblocks.append('### ' + cls.__name__ + '\n')
        subblocks.append(code_snippet(signature))
        docstring = cls.__doc__
        if docstring:
            subblocks.append(process_docstring(docstring))
        blocks.append('\n'.join(subblocks))

    functions = page_data.get('functions', [])
    for module in page_data.get('all_module_functions', []):
        module_functions = []
        for name in dir(module):
            if name[0] == '_' or name in EXCLUDE:
                continue
            module_member = getattr(module, name)
            if inspect.isfunction(module_member):
                function = module_member
                if module.__name__ in function.__module__:
                    if function not in module_functions:
                        module_functions.append(function)
        module_functions.sort(key=lambda x: id(x))
        functions += module_functions

    for function in functions:
        subblocks = []
        signature = get_function_signature(function, method=False)
        signature = signature.replace(function.__module__ + '.', '')
        subblocks.append('### ' + function.__name__ + '\n')
        subblocks.append(code_snippet(signature))
        docstring = function.__doc__
        if docstring:
            subblocks.append(process_docstring(docstring))
        blocks.append('\n\n'.join(subblocks))

    if not blocks:
        raise RuntimeError('Found no content for page ' +
                           page_data['page'])

    mkdown = '\n----\n\n'.join(blocks)
    # save module page.
    # Either insert content into existing page,
    # or create page otherwise
    page_name = page_data['page']
    path = os.path.join('sources', page_name)
    if os.path.exists(path):
        template = open(path).read()
        assert '{{autogenerated}}' in template, ('Template found for ' + path +
                                                 ' but missing {{autogenerated}} tag.')
        mkdown = template.replace('{{autogenerated}}', mkdown)
        print('...inserting autogenerated content into template:', path)
    else:
        print('...creating new page with autogenerated content:', path)
    subdir = os.path.dirname(path)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    open(path, 'w').write(mkdown)

# shutil.copyfile('../CONTRIBUTING.md', 'sources/contributing.md')

    # © 2018 GitHub, Inc.
    # Terms
    # Privacy
