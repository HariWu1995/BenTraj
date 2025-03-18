@echo off
echo Downloading the zip files from Edinburgh pedestrian dataset.

for /F %%d in (days.txt) do (
    echo Downloading %%d.zip...
    curl -o %%d.zip https://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/DAYS/%%d.zip
)

echo Done!