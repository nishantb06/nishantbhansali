---
title: "Pushing New Rows to BigQuery Table in GCP using Go"
date: 2024-01-12T22:26:26+05:30
draft: false
---

In this blog post, we'll explore how to push new rows into a BigQuery table using Go. BigQuery, a serverless and highly-scalable data warehouse, is a part of Google Cloud Platform (GCP). We will be using the **`cloud.google.com/go/bigquery`** package for Go to interact with BigQuery.

## **Introduction**

## **Prerequisites**

Before diving into the code, make sure you have the following set up:

- A GCP project with BigQuery enabled
- Service account credentials in a JSON file
- Go installed on your machine

## **Template**

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"cloud.google.com/go/bigquery"
	"google.golang.org/api/option"
)

// SampleData represents the structure of the data to be inserted into BigQuery.
type SampleData struct {
	Name  string  `bigquery:"name"`   // Name of the data
	Value float32 `bigquery:"value"`  // Value of the data
	Time  string  `bigquery:"time"`   // Time when the data was recorded
}

var credentialsFile = "path-to-credentials.json" // Path to the JSON credentials file

// BatchPushToBigQuery is a function that inserts data into BigQuery in batches.
func BatchPushToBigQuery() {
	ctx := context.Background()

	// Create a new BigQuery client using the provided credentials file
	client, err := bigquery.NewClient(ctx, "your-project-id", option.WithCredentialsFile(credentialsFile))
	if err != nil {
		log.Fatalf("Failed to create BigQuery client: %v", err)
	}

	dataset := client.Dataset("dataset-name") // Specify the dataset name
	table := dataset.Table("table-name")     // Specify the table name

	// Retrieve the schema for the specified table
	meta, err := table.Metadata(ctx)
	if err != nil {
		log.Fatalf("Failed to get table meta data: %v", err)
	}

	// Print the schema of the table
	for _, field := range meta.Schema {
		fmt.Printf("Field Name: %s, Field Type: %s\n", field.Name, field.Type)
	}

	// Create sample data to be inserted into the table
	data := SampleData{
		Name:  "nishant",
		Value: 123.123,
		Time:  time.Now().Format(time.RFC3339),
	}

	data2 := SampleData{
		Name:  "nishant2",
		Value: 123.123,
		Time:  time.Now().Format(time.RFC3339),
	}

	inserter := table.Inserter() // Create an inserter for the table
	items := []*SampleData{
		&data,
		&data2,
	}

	// Insert the data into the table
	if err := inserter.Put(ctx, items); err != nil {
		log.Fatalf("Failed to insert data: %v", err)
	}

	fmt.Println("Data inserted successfully!")
}
```

## **Understanding the Code**

Let's break down the code you've provided:

```go
type SampleData struct {
    Name  string  `bigquery:"name"`   // Name of the data
    Value float32 `bigquery:"value"`  // Value of the data
    Time  string  `bigquery:"time"`   // Time when the data was recorded
}
```

- The **`SampleData`** struct represents the structure of the data to be inserted into BigQuery. The struct tags (**`bigquery:"name"`**, **`bigquery:"value"`**, and **`bigquery:"time"`**) provide metadata to the BigQuery API, mapping the fields of the struct to the corresponding columns in the BigQuery table.

```go
client, err := bigquery.NewClient(ctx, "your-project-id", option.WithCredentialsFile(credentialsFile))
```

- The **`bigquery.NewClient`** function is used to create a new BigQuery client. It takes the context (**`ctx`**), project ID, and options (in this case, the path to the JSON credentials file).

```go
dataset := client.Dataset("dataset-name")
table := dataset.Table("table-name")
```

- Here, we specify the dataset and table within BigQuery where we want to insert the data.

```go
meta, err := table.Metadata(ctx)
```

- We retrieve the schema (metadata) of the specified table. The schema includes information about the columns and their types.

```go
inserter := table.Inserter()
```

- An inserter is created for the table, which will be used to insert data.

```go
data := SampleData{
    Name:  "nishant",
    Value: 123.123,
    Time:  time.Now().Format(time.RFC3339),
}
```

Note that even if the column datatype is `timestamp` in bigquery we have to push a string with proper formatting from our side, which is why we are not using a `time.Time` datatype for this field and using `string` instead

- Sample data is created using the **`SampleData`** struct.

```go
items := []*SampleData{
    &data,
    &data2,
}
```

- An array of data items is created to be inserted into the table.

```go
if err := inserter.Put(ctx, items); err != nil {
    log.Fatalf("Failed to insert data: %v", err)
}
```

- The data is inserted into the specified table using the **`Put`** method of the inserter.

## **Conclusion**

In this blog post, we've covered the basics of pushing new rows into a BigQuery table using Go. Understanding the structure of your data, creating a BigQuery client, and utilising the provided libraries for interaction are crucial steps in achieving successful data insertion. This example can be extended for more complex scenarios, such as streaming data or updating existing records. Make sure to check the official documentation for more details and advanced usage.

