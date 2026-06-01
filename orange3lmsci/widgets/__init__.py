from importlib.resources import files

NAME = "LMSci"
DESCRIPTION = "Orange widgets for Language Model interaction."
ICON = str(files("orange3lmsci") / "icons" / "LMSci-category.svg")
BACKGROUND = "#9FFFBD"

PRIORITY = 3

WIDGETS = [
    'OWLMTask'
]

# The .py file where each widget is implemented
WIDGET_HELP_PATH = (
    # You can link to documentation here
)
