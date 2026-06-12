"""Local, server-less video generation services.

Mirrors :mod:`lity.services.image_generation`: a downloaded model is
loaded in-process (diffusers ``WanPipeline`` on the local GPU, or an MLX runtime)
and a short clip is rendered straight to MP4 — no external server.
"""
