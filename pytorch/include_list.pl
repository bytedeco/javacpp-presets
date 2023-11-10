#!/bin/perl

# Must be run at from javacpp-presets/pytorch after cppbuild.sh has been run 
# for linux-x86_64-gpu

# Generate the lists of includes to parse, in order, from the output
# of g++ -H
# Used to update src/main/resources/org/bytedeco/pytorch/presets/*

use strict;
use warnings;

my %incs;
my @inc_per_depth;

sub flush($) {
    my $min_depth = shift;
    for (my $d = @inc_per_depth - 1; $d >= $min_depth; $d--) {
        if ($inc_per_depth[$d]) {
            foreach my $i (@{$inc_per_depth[$d]}) {
                print "#include \"$i\"\n";
                $incs{$i} = 1;
            }
            undef $inc_per_depth[$d];
        }
    }
}

sub go {
    my $path = join ' ', @_;

    my @inc = `g++ -I. -I torch/csrc/api/include/ -H $path -E 2>&1 > /dev/null`;
    foreach my $i (@inc) {
        chomp $i;
        my ($depth, $f) = $i =~ /^(\.+)\s(.*\.h)$/;
        next unless $depth;
        $depth = length($depth);
        $f =~ s#^\./##;
        next if $f =~ m#^/
  |^ATen/ops/\w+_native\.h$
  |^ATen/ops/\w+_meta\.h$
  |^ATen/ops/\w+_ops\.h$
  |^ATen/ops/_\w+\.h$#x
            or $incs{$f};
        flush($depth);
        my $incs = $inc_per_depth[$depth];
        $incs = $inc_per_depth[$depth] = [] unless $incs;
        push @$incs, $f;
    }
    flush(0);
}

chdir "cppbuild/linux-x86_64-gpu/pytorch/torch/include";

go('torch/csrc/api/include/torch/torch.h', 'torch/script.h');

print <<EOF;

// Included by
// ATen/cudnn/Descriptors.h
// ATen/cudnn/Types.h
// c10/cuda/CUDAGuard.h
EOF

go('ATen/cudnn/Descriptors.h', 'ATen/cudnn/Types.h', 'c10/cuda/CUDAGuard.h', '-I/opt/cuda/targets/x86_64-linux/include/');
