.. _{{ fullname | escape }}:

{{ fullname | escape | underline}}


.. automodule:: {{ fullname }}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Module Attributes') }}

    .. autosummary::
    {% for item in attributes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}
    .. rubric:: {{ _('Functions') }}
    .. testsetup::
        :skipif: skip_doctests

     # import all functions from module since examples don't import them
     from {{ fullname }} import *

    .. doctest::

        # empty test needed in case the module has no example usage.
        # otherwise, testsetup throws an error
        pass

    {% for item in functions %}
    .. autofunction:: {{fullname}}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block classes %}
    {% if classes %}
    .. rubric:: {{ _('Classes') }}
    {% for item in classes %}
    .. autoclass:: {{ fullname }}.{{ item }}
        :members:

    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block exceptions %}
    {% if exceptions %}
    .. rubric:: {{ _('Exceptions') }}

    .. autosummary::
    {% for item in exceptions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
    :toctree:
    :template: autosummary/module.rst
    :recursive:
{% for item in modules %}
    {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
