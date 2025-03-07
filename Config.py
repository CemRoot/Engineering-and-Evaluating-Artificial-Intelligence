class Config:
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'
    TYPE_COLS = ['y2', 'y3', 'y4']  # Dependent variables used for multi-label tasks.
    CLASS_COL = 'y2'               # The primary class column (used to create unified "y").
    GROUPED = 'y1'                 # The column used to group data.
