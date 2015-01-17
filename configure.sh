#!/bin/bash
# Downloads APRIL-ANN, compiles and configures
VERSION=v0.4.1
old_path=$(pwd)
tmp_path=$1

mkdir -p $tmp_path 2> /dev/null

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

error()
{
    echo -e >&2 "${RED}$@${NC}"
    exit 1
}

warning()
{
    echo -e >&2 "${YELLOW}$@${NC}"
}

message()
{
    echo -e >&2 "${GREEN}$@${NC}"
}

check_command()
{
    command -v $1 >/dev/null 2>&1 ||
    error "I require $1 but it's not installed. Aborting."
}

download()
{
    base_url=$1
    file=$2
    data_path=$3
    if [[ ! -e $data_path/$file ]]; then
        echo -e "${GREEN}Downaloading $file${NC}"
        url=$base_url/$file
        wget $url -O $data_path/$file ||
        error "Unable to download from $url"
    else
        echo -e "${YELLOW}Skipping download of $file${NC}"
    fi
}

if [[ -z $APRIL_ANN_TUTORIALS_CONFIGURED ]]; then

    if [ ! $(which april-ann) ]; then

        cd $tmp_path

        if [[ ! -e $VERSION ]]; then
            git clone git@github.com:pakozm/april-ann.git $VERSION
        fi

        cd $VERSION
        warning "You would need to be sudoer to install all APRIL-ANN dependencies"
        ./DEPENDENCIES-INSTALLER.sh || exit 1
        . configure.sh || exit 1

        system=$(uname)
        if [[ $system == "Linux" ]]; then
            make release-atlas || exit 1
        elif [[ $system == "Darwin" ]]; then
            if [ $(which port) ]; then
                make release-macports || exit 1
            elif [ $(which brew) ]; then
                make release-homebrew
            else
                error "Needs macports or homebrew installed"
            fi
        else
            error "Not supported system: $system"
            exit 1
        fi

        cd $old_path

    else
        warning "april-ann executable in PATH, skipping installation"
    fi
    export APRIL_ANN_TUTORIALS_CONFIGURED=1
fi
