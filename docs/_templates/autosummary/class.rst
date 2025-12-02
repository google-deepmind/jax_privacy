{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
   :special-members: __call__

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      {% for item in methods %}
         ~{{ name }}.{{ item }}
      {% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:
      {% for item in attributes %}
         ~{{ name }}.{{ item }}
      {% endfor %}
   {% endif %}
   {% endblock %}
