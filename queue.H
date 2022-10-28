/*
 File: queue.H

 Author:
 Date  :

 */

#ifndef QUEUE_H
#define QUEUE_H

#include "thread.H"

class Queue
{
private:
    Thread *thread;
    Queue *next;

public:
    Queue()
    {
        thread = NULL;
        next = NULL;
    }

    Queue(Thread *thread_new)
    {
        thread = thread_new;
        next = NULL;
    }

    void enqueue(Thread *thread_new)
    {
        if (thread == NULL)
        {
            thread = thread_new;
        }
        else
        {
            if (next == NULL)
            {
                next = new Queue(thread_new);
            }
            else
            {
                next->enqueue(thread_new);
            }
        }
    }

    Thread *dequeue()
    {
        if (thread == NULL)
            return NULL;

        Thread *thread_top = thread;

        if (next != NULL)
        {
            thread = next->thread;
            Queue *stale_next = next;
            next = next->next;

            delete stale_next;
        }
        else
        {
            thread = NULL;
        }

        return thread_top;
    }
};

#endif
