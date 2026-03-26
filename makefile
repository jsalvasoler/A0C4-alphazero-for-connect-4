fmt:  # Format and check code with ruff
	uv tool run ruff format
	uv tool run ruff check --fix
