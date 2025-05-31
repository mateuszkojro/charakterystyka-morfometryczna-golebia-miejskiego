for nb in *.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb" --inplace
done