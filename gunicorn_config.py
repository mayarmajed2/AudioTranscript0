# gunicorn configuration file
bind = "0.0.0.0:5000"
reload = True
reuse_port = True

# For handling large uploads
timeout = 300  # Increased timeout to 5 minutes
workers = 2
threads = 4
worker_class = "sync"
worker_connections = 1000
keepalive = 5

# Limit request sizes better than Flask's MAX_CONTENT_LENGTH
limit_request_line = 0  # Disable line length checking
limit_request_fields = 0  # Disable field count checking
limit_request_field_size = 0  # Disable field size checking

# Logging
loglevel = "debug"
accesslog = "-"
errorlog = "-"