foreach ($nb in Get-ChildItem -Filter "*.ipynb") {
    jupyter nbconvert --to notebook --execute $nb.FullName --inplace
}
