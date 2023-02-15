import gin
import tensorflow as tf
import logging
import wandb
import numpy as np
import matplotlib.pyplot as plt
@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, model_name, total_steps, log_interval, ckpt_interval, learning_rate, patience, model_save_dir):
        # Summary Writer
        self.train_writer = tf.summary.create_file_writer(run_paths['path_summary_train'])
        self.val_writer = tf.summary.create_file_writer(run_paths['path_summary_val'])
        
        # Loss objective
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 10000, 0.9)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate= self.lr)
        self.model = model
        self.model_name  =  model_name
        self.model_save_dir = model_save_dir

        # Checkpoint Manager
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory= run_paths["path_ckpts_train"], max_to_keep=5)

        #best_weights_checkpoint
        self.bst_checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.bst_manager = tf.train.CheckpointManager(self.bst_checkpoint, directory="/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/checkpoints/s2l/best_ckpt/"+ model_name, max_to_keep=1)
        
        #Early Stopping Metrics
        self.wait =1
        self.patience = patience

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

        #Initialize training data
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        model.summary()    
    
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss =self.loss_object(labels,predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)
        


    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels,predictions)
        
        self.val_loss.update_state(t_loss)
        self.val_accuracy.update_state(labels, predictions)

    def train(self):
        logging.info('Training Initiated..............')
        best_val_acc = 0
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                #Early Stopping
                if self.val_accuracy.result().numpy() * 100 > best_val_acc:
                        self.wait= 1
                        best_val_acc = self.val_accuracy.result().numpy() * 100 
                        logging.info(f' New best found....Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                        # Save checkpoint
                        self.bst_manager.save(checkpoint_number=step, check_interval=True)
                        #Save model only when accuracy is greater than 85 percent
                        if self.val_accuracy.result().numpy() * 100 > 85:
                            tf.get_logger().setLevel('ERROR')
                            self.model.save(self.model_save_dir + self.model_name)
                        
                else:
                    self.wait+=1
                    #If val acccuracy does not improve for a predefined number of times (wait) Training is stopped
                    if self.wait > self.patience:
                        logging.info(f'Early Stopping training after {step} steps.')
                        
                        # Save final checkpoint
                        self.manager.save(checkpoint_number=step)
                        logging.info(f'Finished training after {step} steps.')
                        return self.val_accuracy.result().numpy()
                
                # Write summary to tensorboard
                log_data = {"Step" : step, "Loss": self.train_loss.result(), "Accuracy": self.train_accuracy.result() * 100,
                            "Validation Loss": self.val_loss.result(), "Validation Accuracy": self.val_accuracy.result() * 100 }
                wandb.log(log_data)

                #Write the summary to Summary writer
                with self.train_writer.as_default():
                    tf.summary.scalar('train_loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)

                with self.val_writer.as_default():
                    tf.summary.scalar('val_loss', self.val_loss.result(), step=step)
                    tf.summary.scalar('val_accuracy', self.val_accuracy.result(), step=step)
                
                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.manager.save(checkpoint_number=step, check_interval=True)

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                tf.get_logger().setLevel('ERROR')
                self.manager.save(checkpoint_number=step)
                self.model.save(self.model_save_dir + self.model_name)
                return self.val_accuracy.result().numpy()
