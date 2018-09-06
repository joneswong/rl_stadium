from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


FLAGS = tf.flags.FLAGS
# pai tf used
tf.flags.DEFINE_string("ps_hosts", "", "ps_hosts")
tf.flags.DEFINE_string("worker_hosts", "", "worker_hosts")
tf.flags.DEFINE_string("job_name", "", "job_name")
tf.flags.DEFINE_integer("task_index", "-1", "task_index")
tf.flags.DEFINE_string("tables", "", "tables names")
tf.flags.DEFINE_string("outputs", "", "output tables names")
tf.flags.DEFINE_string("checkpointDir", "", "oss buckets for saving checkpoint")
tf.flags.DEFINE_string("buckets", "", "oss buckets")

def main(_):
    # distributed pai tf
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    
    shared_job_device = "/job:ps/task:0"
    local_job_device = "/job:" + FLAGS.job_name + ("/task:%d" % FLAGS.task_index)
    global_variable_device = shared_job_device + "/cpu"

    is_learner = (FLAGS.job_name == "ps")

    with tf.device(global_variable_device):
        a = tf.get_variable(name='a', dtype=tf.float32, shape=())
        tf.add_to_collection(tf.GraphKeys.INIT_OP, a.initializer)

    with tf.device(local_job_device):
        if not is_learner:
            b = tf.get_variable(name='b', dtype=tf.float32, shape=())
            op = tf.assign(b, a)
            tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, b.initializer)

    #tf.add_to_collection(tf.GraphKeys.READY_OP, tf.report_uninitialized_variables([a, b]))
    tf.add_to_collection(tf.GraphKeys.READY_FOR_LOCAL_INIT_OP, tf.report_uninitialized_variables([a]))
    
    # create session and run
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.MonitoredTrainingSession(
        server.target,
        is_chief=is_learner,
        checkpoint_dir=FLAGS.checkpointDir,
        save_checkpoint_secs=600,
        save_summaries_secs=30,
        log_step_count_steps=50000,
        config=config) as session:
        print("***************hello world******************")
        if is_learner:
            print(session.run(a))
        else:
            print(session.run(op))
        while True:
            pass

    print("done.")


if __name__ == "__main__":
    tf.app.run()
