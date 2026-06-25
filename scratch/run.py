""" Run """

from farms_core import pylog
from farms_app.core.application import FARMSApplication
from farms_app.core.options import ApplicationOptions

pylog.set_level("debug")


app_options = ApplicationOptions(title="FARMS-APP")

app = FARMSApplication.from_options(app_options)
app.run()
