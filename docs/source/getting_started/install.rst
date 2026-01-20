============
Installation
============

The standard installation installs the CPU version of ERBS. To enable GPU
support (available only on **Linux**), install ERBS with the extra CUDA
dependencies. See the
`Jax installation instructions <https://github.com/google/jax#installation>`_
for more details.

From PyPI
---------

**CPU:**

.. highlight:: bash
.. code-block:: bash

    pip install erbs

**GPU:**

.. highlight:: bash
.. code-block:: bash

    pip install "erbs[cuda]"

From GitHub
-----------

For a pre-release version, install ERBS directly from GitHub.

**CPU:**

.. highlight:: bash
.. code-block:: bash

    pip install git+https://github.com/apax-hub/erbs.git

**GPU:**

.. highlight:: bash
.. code-block:: bash

    pip install erbs[cuda] git+https://github.com/apax-hub/erbs.git

For Developers
--------------

To set up a development environment, first install `uv`_.

.. highlight:: bash
.. code-block:: bash

    pip install uv


Then clone the project from GitHub,

.. highlight:: bash
.. code-block:: bash

    git clone https://github.com/apax-hub/erbs.git <dest_dir>
    cd <dest_dir>

and install it.

**CPU:**

.. highlight:: bash
.. code-block:: bash

    uv sync --all-extras --no-extra cuda

**GPU:**

.. highlight:: bash
.. code-block:: bash

    uv sync --extra cuda

Extra Dependencies
------------------

If you want to use ERBS in the Zntrack/IPSuite framework and use the predefined
`erbs.nodes`, you can install the extra dependencies for Zntrack:

.. highlight:: bash
.. code-block:: bash

    pip install "erbs[zntrack]"


.. _uv: https://astral.sh/blog/uv
