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
                print "#include \"$i\"\n" unless $incs{$i};
                $incs{$i} = 1;
            }
            undef $inc_per_depth[$d];
        }
    }
}

sub go {
    my ($roots, $opts) = @_;
    my $path = join ' ', @$roots, @$opts;

    my $exe = "g++ -I. -I torch/csrc/api/include/ -DUSE_UCC -DUSE_C10D_GLOO -DUSE_C10D_MPI -DUSE_DISTRIBUTED -H $path -E 2>&1 > /dev/null";
    #my $exe = "g++ -I. -I torch/csrc/api/include/ -DUSE_UCC -DUSE_C10D_GLOO -DUSE_C10D_MPI -DUSE_DISTRIBUTED -D_WIN32 -H $path -E 2>&1 > /dev/null";
    my @inc = `$exe`;
    if ($? != 0) {
      print STDERR "Failed:\n$exe\nError: $?: $!\n";
      exit $?;
    }

    foreach my $i (@inc) {
        chomp $i;
        my ($depth, $f) = $i =~ /^(\.+)\s(.*\.h(?:pp)?)$/;
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
    foreach my $i (@$roots) {
      print "#include \"$i\"\n" unless $incs{$i};
      $incs{$i} = 1;
    }
}

chdir "cppbuild/linux-x86_64-gpu/pytorch/torch/include";

print <<EOF;
// Included by
// torch/csrc/api/include/torch/torch.h
// torch/script.h
// torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h
// torch/csrc/distributed/c10d/ProcessGroupGloo.hpp
// torch/csrc/distributed/c10d/PrefixStore.hpp
// torch/csrc/distributed/c10d/logger.hpp
EOF

go(['torch/csrc/api/include/torch/torch.h', 'torch/script.h', 'torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h', 'torch/csrc/distributed/c10d/ProcessGroupGloo.hpp', 'torch/csrc/distributed/c10d/PrefixStore.hpp', 'torch/csrc/distributed/c10d/logger.hpp'], []);

print <<EOF;

// Included by
// ATen/cudnn/Types.h
// ATen/cudnn/Descriptors.h
// ATen/cuda/CUDAEvent.h
// torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h
EOF

go(['ATen/cudnn/Types.h', 'ATen/cudnn/Descriptors.h', 'ATen/cuda/CUDAEvent.h', 'torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h'], ['-I/opt/cuda/targets/x86_64-linux/include/', '-DUSE_CUDA']);
