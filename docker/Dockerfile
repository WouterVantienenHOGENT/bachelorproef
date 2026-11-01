FROM debian:bookworm

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-lang-european \
    texlive-bibtex-extra \
    biber \
    git \
    texlive-xetex \
    python3-pygments \
    fontconfig \
    latexmk \
    && rm -rf /var/lib/apt/lists/*
COPY fonts/ /usr/share/fonts/opentype
RUN fc-cache -f -v
